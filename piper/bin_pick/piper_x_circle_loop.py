from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os

import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.prims import GeometryPrim, SingleArticulation
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ISAAC_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
DEFAULT_PIPER_USD = os.path.join(ISAAC_ROOT, "piper_isaac_sim", "USD", "piper_x_v1.usd")
DEFAULT_LULA_YAML = os.path.join(REPO_ROOT, "desc", "piper_x_robot_description.yaml")
DEFAULT_LULA_URDF = os.path.join(
    ISAAC_ROOT, "piper_isaac_sim", "piper_x_description", "urdf", "piper_x_description_d435.urdf"
)

parser = argparse.ArgumentParser()
parser.add_argument("--robot-usd", type=str, default=DEFAULT_PIPER_USD)
parser.add_argument("--lula-yaml", type=str, default=DEFAULT_LULA_YAML)
parser.add_argument("--lula-urdf", type=str, default=DEFAULT_LULA_URDF)
parser.add_argument("--ee-frame", type=str, default="Link6")
parser.add_argument("--robot-x", type=float, default=0.0)
parser.add_argument("--robot-y", type=float, default=0.12)
parser.add_argument("--robot-yaw", type=float, default=90.0)
parser.add_argument("--target-x", type=float, default=0.0)
parser.add_argument("--target-y", type=float, default=0.3)
parser.add_argument("--target-z", type=float, default=1.1)
parser.add_argument("--roll", type=float, default=-60.0)
parser.add_argument("--pitch", type=float, default=0.0)
parser.add_argument("--yaw", type=float, default=0.0)
parser.add_argument("--radius", type=float, default=0.1)
parser.add_argument("--num-points", type=int, default=48)
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


def create_omnipbr_material(stage, material_path, color=None, roughness=0.5, texture_path=None):
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
    omni.usd.create_material_input(material_prim, "reflection_roughness_constant", roughness, Sdf.ValueTypeNames.Float)
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
    }
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_prim_path + "/") and prim.GetName().lower() in candidate_tokens:
            prim.SetActive(False)


def set_translate(prim, translate):
    xformable = UsdGeom.Xformable(prim)
    ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    vec = Gf.Vec3d(float(translate[0]), float(translate[1]), float(translate[2]))
    if ops:
        ops[0].Set(vec)
    else:
        xformable.AddTranslateOp().Set(vec)


def create_marker(stage, prim_path, radius=0.02, color=(0.88, 0.18, 0.16)):
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.CreateRadiusAttr(radius)
    UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
    material = create_omnipbr_material(stage, f"{prim_path}_Looks", color=color, roughness=0.2)
    bind_material(sphere.GetPrim(), material)
    return sphere.GetPrim()


world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Table")
UsdGeom.Xform.Define(stage, "/World/Bin")
UsdGeom.Xform.Define(stage, "/World/Lights")
UsdGeom.Xform.Define(stage, "/World/RobotMount")
UsdGeom.Xform.Define(stage, "/World/Debug")

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
bin_material = create_omnipbr_material(stage, "/World/Looks/Bin", color=(0.24, 0.24, 0.26), roughness=0.45)

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

add_reference_to_stage(usd_path=args.robot_usd, prim_path="/World/RobotMount/PiperX/Robot")
deactivate_embedded_environment(stage, "/World/RobotMount/PiperX/Robot")
robot = SingleArticulation(prim_path="/World/RobotMount/PiperX/Robot", name="piper_x")

set_camera_view(
    eye=[1.7, -1.6, 1.45],
    target=[0.0, 0.15, 0.88],
    camera_prim_path="/OmniverseKit_Persp",
)

world.reset()
robot.initialize()

lula_solver = LulaKinematicsSolver(robot_description_path=args.lula_yaml, urdf_path=args.lula_urdf)
robot_base_position = np.array([args.robot_x, args.robot_y, table_surface_z], dtype=np.float64)
robot_base_orientation = euler_angles_to_quats(np.array([0.0, 0.0, args.robot_yaw]), degrees=True)
lula_solver.set_robot_base_pose(robot_base_position, robot_base_orientation)
art_kinematics = ArticulationKinematicsSolver(robot, lula_solver, args.ee_frame)
articulation_controller = robot.get_articulation_controller()

_, current_rotation = art_kinematics.compute_end_effector_pose()
base_orientation = rot_matrices_to_quats(np.asarray(current_rotation))
orientation_offset = euler_angles_to_quats(np.array([args.roll, args.pitch, args.yaw]), degrees=True)
target_orientation_quat = Gf.Quatf(float(base_orientation[0]), Gf.Vec3f(*base_orientation[1:])) * Gf.Quatf(
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

center = np.array([args.target_x, args.target_y, args.target_z], dtype=np.float64)
angles = np.linspace(0.0, -2.0 * np.pi, args.num_points, endpoint=False)
circle_targets = [center + np.array([args.radius * np.cos(theta), args.radius * np.sin(theta), 0.0]) for theta in angles]

target_marker = create_marker(stage, "/World/Debug/CircleTarget", radius=0.018, color=(0.88, 0.18, 0.16))
point_index = 0
last_printed_index = None

print(f"[PiperCircle] center: {center.tolist()}")
print(f"[PiperCircle] orientation offset deg: {[args.roll, args.pitch, args.yaw]}")
print(f"[PiperCircle] radius: {args.radius}")
print(f"[PiperCircle] num points: {args.num_points}")

while simulation_app.is_running():
    target_position = circle_targets[point_index]
    set_translate(target_marker, target_position)

    action, success = art_kinematics.compute_inverse_kinematics(
        target_position=target_position,
        target_orientation=target_orientation,
    )

    if success:
        articulation_controller.apply_action(action)
        point_index = (point_index + 1) % len(circle_targets)
        last_printed_index = None
    elif last_printed_index != point_index:
        print(f"[PiperCircle] IK failed at point {point_index}: {target_position.tolist()}")
        last_printed_index = point_index

    world.step(render=True)

simulation_app.close()
