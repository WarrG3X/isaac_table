from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os

import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import GeometryPrim, SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ISAAC_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
DEFAULT_PIPER_USD = os.path.join(ISAAC_ROOT, "piper_isaac_sim", "USD", "piper_x_v1.usd")


parser = argparse.ArgumentParser()
parser.add_argument("--robot-usd", type=str, default=DEFAULT_PIPER_USD)
parser.add_argument("--robot-x", type=float, default=0.0)
parser.add_argument("--robot-y", type=float, default=-0.18)
parser.add_argument("--robot-yaw", type=float, default=90.0)
parser.add_argument("--table-width", type=float, default=1.40)
parser.add_argument("--table-depth", type=float, default=1.55)
parser.add_argument("--cube-size", type=float, default=0.05)
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


def create_omnipbr_material(stage, material_path, color=None, roughness=0.5, metallic=0.0, texture_path=None):
    created = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=created,
    )
    material_prim = stage.GetPrimAtPath(created[0])
    if created[0] != material_path:
        omni.kit.commands.execute("MovePrim", path_from=created[0], path_to=material_path)
        material_prim = stage.GetPrimAtPath(material_path)
    if color is not None:
        omni.usd.create_material_input(
            material_prim, "diffuse_color_constant", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f
        )
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
        "data",
    }
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_prim_path + "/") and prim.GetName().lower() in candidate_tokens:
            prim.SetActive(False)


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

bamboo_texture_path = os.path.join(REPO_ROOT, "materials", "nv_bamboo_desktop.jpg")
granite_texture_path = os.path.join(REPO_ROOT, "materials", "nv_granite_tile.jpg")

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
source_pad_material = create_omnipbr_material(stage, "/World/Looks/SourcePad", color=(0.78, 0.16, 0.16), roughness=0.42)
target_pad_material = create_omnipbr_material(stage, "/World/Looks/TargetPad", color=(0.16, 0.28, 0.82), roughness=0.42)

floor_prim = define_box(stage, "/World/Floor/Base", size=(7.0, 7.0, 0.02), translate=(0.0, 0.0, -0.01))
bind_material(floor_prim, floor_material_base)
floor_surface = define_uv_plane(stage, "/World/Floor/Surface", size=(6.92, 6.92), translate=(0.0, 0.0, 0.001), uv_scale=(6.0, 6.0))
bind_material(floor_surface, floor_material)

table_size = (args.table_width, args.table_depth, 0.06)
table_top_z = 0.75
table_surface_z = table_top_z + table_size[2] * 0.5
top_prim = define_box(stage, "/World/Table/Top", size=table_size, translate=(0.0, 0.0, table_top_z))
bind_material(top_prim, wood_base_material)
table_surface = define_uv_plane(
    stage,
    "/World/Table/TopSurface",
    size=(table_size[0] - 0.04, table_size[1] - 0.04),
    translate=(0.0, 0.0, table_surface_z + 0.001),
)
bind_material(table_surface, wood_material)

leg_height = 0.72
leg_center_z = leg_height * 0.5
leg_dx = table_size[0] * 0.5 - 0.08
leg_dy = table_size[1] * 0.5 - 0.08
table_collision_paths = [str(floor_prim.GetPath()), str(top_prim.GetPath())]
for name, position in {
    "LegFL": (leg_dx, leg_dy, leg_center_z),
    "LegFR": (leg_dx, -leg_dy, leg_center_z),
    "LegBL": (-leg_dx, leg_dy, leg_center_z),
    "LegBR": (-leg_dx, -leg_dy, leg_center_z),
}.items():
    leg_prim = define_box(stage, f"/World/Table/{name}", size=(0.06, 0.06, leg_height), translate=position)
    bind_material(leg_prim, leg_material)
    table_collision_paths.append(str(leg_prim.GetPath()))

pad_size = (0.18, 0.18)
pad_z = table_surface_z + 0.002
pad_y = 0.22
source_pad_center = (-0.16, pad_y, pad_z)
target_pad_center = (0.16, pad_y, pad_z)
source_pad = define_uv_plane(stage, "/World/SourcePad/Plane", size=pad_size, translate=source_pad_center)
bind_material(source_pad, source_pad_material)
target_pad = define_uv_plane(stage, "/World/TargetPad/Plane", size=pad_size, translate=target_pad_center)
bind_material(target_pad, target_pad_material)

apply_static_collision(table_collision_paths)

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

mount_prim = UsdGeom.Xform.Define(stage, "/World/RobotMount/PiperX")
mount_xform = UsdGeom.Xformable(mount_prim.GetPrim())
mount_xform.AddTranslateOp().Set(Gf.Vec3d(args.robot_x, args.robot_y, table_surface_z))
mount_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, args.robot_yaw))
add_reference_to_stage(usd_path=os.path.abspath(args.robot_usd), prim_path="/World/RobotMount/PiperX/Robot")
deactivate_embedded_environment(stage, "/World/RobotMount/PiperX/Robot")

cube_center = np.array(
    [
        source_pad_center[0],
        source_pad_center[1],
        table_surface_z + args.cube_size * 0.5 + 0.002,
    ],
    dtype=np.float32,
)
cube = DynamicCuboid(
    prim_path="/World/ObjectCube",
    name="simple_pick_cube",
    position=cube_center,
    scale=np.array([args.cube_size, args.cube_size, args.cube_size], dtype=np.float32),
    color=np.array([0.18, 0.76, 0.28], dtype=np.float32),
)
world.scene.add(cube)

robot = SingleArticulation("/World/RobotMount/PiperX/Robot")
world.scene.add(robot)

set_camera_view(
    eye=[1.9, -1.55, 1.45],
    target=[0.0, 0.32, 0.86],
    camera_prim_path="/OmniverseKit_Persp",
)

world.reset()
robot.initialize()

print(f"[SimplePick] source pad center: {source_pad_center}")
print(f"[SimplePick] target pad center: {target_pad_center}")
print(f"[SimplePick] cube start: {cube_center.tolist()}")
print("[SimplePick] success condition: cube center inside blue square bounds")

target_half_extent_x = pad_size[0] * 0.5
target_half_extent_y = pad_size[1] * 0.5

while simulation_app.is_running():
    world.step(render=True)
    if not world.is_playing():
        continue
    cube_pos = np.asarray(cube.get_world_pose()[0], dtype=np.float32)
    in_target = (
        abs(float(cube_pos[0] - target_pad_center[0])) <= target_half_extent_x
        and abs(float(cube_pos[1] - target_pad_center[1])) <= target_half_extent_y
        and float(cube_pos[2]) <= table_surface_z + args.cube_size
    )
    if in_target:
        print(f"[SimplePick] success: cube entered target square at {cube_pos}")
        break

simulation_app.close()
