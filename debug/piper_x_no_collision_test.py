from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os

import numpy as np
import isaacsim.core.utils.numpy.rotations as rot_utils
import omni.kit.commands
import omni.ui as ui
from isaacsim.core.api import World
from isaacsim.core.prims import GeometryPrim, SingleArticulation
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ISAAC_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
DEFAULT_PIPER_USD = os.path.join(ISAAC_ROOT, "piper_isaac_sim", "USD", "piper_x_v1.usd")
DEFAULT_LULA_YAML = os.path.join(REPO_ROOT, "desc", "piper_x_robot_description.yaml")
DEFAULT_LULA_URDF = os.path.join(
    ISAAC_ROOT, "piper_isaac_sim", "piper_x_description", "urdf", "piper_x_description_d435.urdf"
)


parser = argparse.ArgumentParser()
parser.add_argument("--robot-usd", type=str, default=DEFAULT_PIPER_USD)
parser.add_argument("--robot-x", type=float, default=0.0)
parser.add_argument("--robot-y", type=float, default=0.12)
parser.add_argument("--robot-yaw", type=float, default=90.0)
parser.add_argument("--lula-yaml", type=str, default=DEFAULT_LULA_YAML)
parser.add_argument("--lula-urdf", type=str, default=DEFAULT_LULA_URDF)
parser.add_argument("--ee-frame", type=str, default="Link6")
args, _ = parser.parse_known_args()


def define_box(stage, prim_path, size, translate):
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*translate))
    cube = UsdGeom.Cube.Define(stage, f"{prim_path}/Geom")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*size))
    return cube.GetPrim()


def create_omnipbr_material(stage, material_path, color, roughness=0.5):
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
    omni.usd.create_material_input(
        material_prim, "diffuse_color_constant", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f
    )
    omni.usd.create_material_input(
        material_prim, "reflection_roughness_constant", float(roughness), Sdf.ValueTypeNames.Float
    )
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


def disable_robot_collisions(stage, root_prim_path):
    disabled_api_count = 0
    deactivated_branch_count = 0
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if not prim_path.startswith(root_prim_path + "/"):
            continue
        lower_path = prim_path.lower()
        if "/collisions/" in lower_path or lower_path.endswith("/collisions") or "/collision/" in lower_path or lower_path.endswith("/collision"):
            prim.SetActive(False)
            deactivated_branch_count += 1
            continue
        collision_api = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath())
        if collision_api:
            attr = collision_api.GetCollisionEnabledAttr()
            if attr:
                attr.Set(False)
                disabled_api_count += 1
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Get(stage, prim.GetPath())
        if physx_collision_api:
            attr = physx_collision_api.GetCollisionEnabledAttr()
            if attr:
                attr.Set(False)
                disabled_api_count += 1
    print(
        f"[NoCollision] deactivated collision branches={deactivated_branch_count} "
        f"disabled collision APIs={disabled_api_count}"
    )


def build_joint_window(joint_names, lower_limits, upper_limits, initial_positions, open_gripper_fn=None, close_gripper_fn=None):
    joint_models = {}
    window = ui.Window("Piper Joint Control", width=460, height=360, visible=True)
    window.position_x = 40
    window.position_y = 80
    with window.frame:
        with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
            with ui.VStack(spacing=8, height=0):
                ui.Label("Manual Joint Targets")
                if open_gripper_fn is not None and close_gripper_fn is not None:
                    with ui.HStack(height=28):
                        ui.Button("Open Gripper", clicked_fn=open_gripper_fn)
                        ui.Button("Close Gripper", clicked_fn=close_gripper_fn)
                for idx, joint_name in enumerate(joint_names):
                    model = ui.SimpleFloatModel(float(initial_positions[idx]))
                    joint_models[joint_name] = model
                    with ui.HStack(height=24):
                        ui.Label(joint_name, width=70)
                        ui.FloatDrag(model=model, min=float(lower_limits[idx]), max=float(upper_limits[idx]), width=80)
                        ui.FloatSlider(model=model, min=float(lower_limits[idx]), max=float(upper_limits[idx]))
    return window, joint_models


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
    window = ui.Window("No-Collision Lula IK", width=380, height=240, visible=True)
    window.position_x = 520
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


def create_target_marker(stage, prim_path):
    marker = UsdGeom.Sphere.Define(stage, prim_path)
    marker.CreateRadiusAttr(0.025)
    return marker.GetPrim()


def set_translate(prim, translate):
    tx, ty, tz = float(translate[0]), float(translate[1]), float(translate[2])
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(tx, ty, tz))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))


def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Floor")
UsdGeom.Xform.Define(stage, "/World/Table")
UsdGeom.Xform.Define(stage, "/World/Bin")
UsdGeom.Xform.Define(stage, "/World/Lights")
UsdGeom.Xform.Define(stage, "/World/RobotMount")

floor_material = create_omnipbr_material(stage, "/World/Looks/Floor", color=(0.74, 0.76, 0.78), roughness=0.7)
table_material = create_omnipbr_material(stage, "/World/Looks/Table", color=(0.62, 0.56, 0.46), roughness=0.42)
leg_material = create_omnipbr_material(stage, "/World/Looks/Legs", color=(0.20, 0.14, 0.11), roughness=0.5)
bin_material = create_omnipbr_material(stage, "/World/Looks/Bin", color=(0.22, 0.22, 0.24), roughness=0.35)

floor_prim = define_box(stage, "/World/Floor/Base", size=(6.8, 6.8, 0.02), translate=(0.0, 0.0, -0.01))
bind_material(floor_prim, floor_material)

table_top_z = 0.75
table_surface_z = table_top_z + 0.03
top_prim = define_box(stage, "/World/Table/Top", size=(1.4, 0.8, 0.06), translate=(0.0, 0.0, table_top_z))
bind_material(top_prim, table_material)

table_leg_paths = []
for name, position in {
    "LegFL": (0.62, 0.32, 0.36),
    "LegFR": (0.62, -0.32, 0.36),
    "LegBL": (-0.62, 0.32, 0.36),
    "LegBR": (-0.62, -0.32, 0.36),
}.items():
    leg_prim = define_box(stage, f"/World/Table/{name}", size=(0.06, 0.06, 0.72), translate=position)
    bind_material(leg_prim, leg_material)
    table_leg_paths.append(str(leg_prim.GetPath()))

table_half_depth = 0.4
bin_outer_width = 0.72
bin_outer_depth = 0.48
bin_height = 0.36
bin_wall = 0.025
bin_center_z = table_surface_z - bin_height * 0.5
bin_center_y = table_half_depth + bin_outer_depth * 0.5 - bin_wall * 0.35
for name, (size, position) in {
    "Bottom": ((bin_outer_width, bin_outer_depth, bin_wall), (0.0, bin_center_y, table_surface_z - bin_height + bin_wall * 0.5)),
    "WallLeft": ((bin_wall, bin_outer_depth, bin_height), (-bin_outer_width * 0.5 + bin_wall * 0.5, bin_center_y, bin_center_z)),
    "WallRight": ((bin_wall, bin_outer_depth, bin_height), (bin_outer_width * 0.5 - bin_wall * 0.5, bin_center_y, bin_center_z)),
    "WallInner": ((bin_outer_width - 2 * bin_wall, bin_wall, bin_height), (0.0, bin_center_y - bin_outer_depth * 0.5 + bin_wall * 0.5, bin_center_z)),
    "WallOuter": ((bin_outer_width - 2 * bin_wall, bin_wall, bin_height), (0.0, bin_center_y + bin_outer_depth * 0.5 - bin_wall * 0.5, bin_center_z)),
}.items():
    bin_prim = define_box(stage, f"/World/Bin/{name}", size=size, translate=position)
    bind_material(bin_prim, bin_material)

apply_static_collision([str(floor_prim.GetPath()), str(top_prim.GetPath()), *table_leg_paths])

dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
dome_light.CreateIntensityAttr(900.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))

sun_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Sun"))
sun_light.CreateIntensityAttr(3200.0)
sun_light.CreateAngleAttr(0.6)
sun_light.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
UsdGeom.Xformable(sun_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))

mount_prim = UsdGeom.Xform.Define(stage, "/World/RobotMount/PiperX")
mount_xform = UsdGeom.Xformable(mount_prim.GetPrim())
mount_xform.AddTranslateOp().Set(Gf.Vec3d(args.robot_x, args.robot_y, table_surface_z))
mount_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, args.robot_yaw))
add_reference_to_stage(usd_path=os.path.abspath(args.robot_usd), prim_path="/World/RobotMount/PiperX/Robot")
deactivate_embedded_environment(stage, "/World/RobotMount/PiperX/Robot")
disable_robot_collisions(stage, "/World/RobotMount/PiperX/Robot")

set_camera_view(
    eye=[1.30, -1.15, 1.15],
    target=[0.05, 0.12, 0.83],
    camera_prim_path="/OmniverseKit_Persp",
)

robot = SingleArticulation("/World/RobotMount/PiperX/Robot")
world.scene.add(robot)
world.reset()
robot.initialize()

joint_names = robot.dof_names
joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
limits = robot._articulation_view.get_dof_limits()[0]
lower_limits = limits[:, 0]
upper_limits = limits[:, 1]
initial_positions = robot.get_joint_positions()
dof_props = robot.dof_properties

gripper_open_values = {
    "joint7": float(upper_limits[joint_name_to_index["joint7"]]),
    "joint8": float(lower_limits[joint_name_to_index["joint8"]]),
}
gripper_closed_values = {
    "joint7": float(lower_limits[joint_name_to_index["joint7"]]),
    "joint8": float(upper_limits[joint_name_to_index["joint8"]]),
}


def set_gripper(open_state: bool):
    target_values = gripper_open_values if open_state else gripper_closed_values
    joint_models["joint7"].set_value(target_values["joint7"])
    joint_models["joint8"].set_value(target_values["joint8"])

print(f"[NoCollision] articulation prim: /World/RobotMount/PiperX/Robot")
print(f"[NoCollision] dof count: {len(joint_names)}")
print(f"[NoCollision] joints: {joint_names}")
for gripper_joint in ("joint7", "joint8"):
    idx = joint_name_to_index[gripper_joint]
    print(
        f"[NoCollision] {gripper_joint} props: "
        f"lower={dof_props['lower'][idx]:.5f} upper={dof_props['upper'][idx]:.5f} "
        f"stiffness={dof_props['stiffness'][idx]:.3f} damping={dof_props['damping'][idx]:.3f} "
        f"maxEffort={dof_props['maxEffort'][idx]:.3f} maxVelocity={dof_props['maxVelocity'][idx]:.3f}"
    )

joint_window, joint_models = build_joint_window(
    joint_names,
    lower_limits,
    upper_limits,
    initial_positions,
    open_gripper_fn=lambda: set_gripper(True),
    close_gripper_fn=lambda: set_gripper(False),
)

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
print("[NoCollision] applied gripper hold override on joint7/8")

lula_solver = LulaKinematicsSolver(robot_description_path=os.path.abspath(args.lula_yaml), urdf_path=os.path.abspath(args.lula_urdf))
robot_base_position = np.array([args.robot_x, args.robot_y, table_surface_z], dtype=np.float64)
robot_base_orientation = euler_angles_to_quats(np.array([0.0, 0.0, args.robot_yaw]), degrees=True)
lula_solver.set_robot_base_pose(robot_base_position, robot_base_orientation)
art_kinematics = ArticulationKinematicsSolver(robot, lula_solver, args.ee_frame)
ee_position, ee_rotation = art_kinematics.compute_end_effector_pose()
ik_base_position = np.array(ee_position, dtype=np.float32)
ik_base_orientation = rot_matrices_to_quats(np.asarray(ee_rotation))
print(f"[NoCollision] Lula YAML: {os.path.abspath(args.lula_yaml)}")
print(f"[NoCollision] Lula URDF: {os.path.abspath(args.lula_urdf)}")
print(f"[NoCollision] EE frame: {args.ee_frame}")
print(f"[NoCollision] current ee position: {ee_position}")
print(f"[NoCollision] Lula active joints: {lula_solver.get_joint_names()}")

ik_target_marker = create_target_marker(stage, "/World/Debug/IkTarget")
target_material = create_omnipbr_material(stage, "/World/Looks/IkTarget", color=(0.82, 0.16, 0.12), roughness=0.25)
bind_material(ik_target_marker, target_material)
set_translate(ik_target_marker, ik_base_position)
ik_window, ik_models, ik_status = build_ik_window(ik_base_position)
last_valid_arm_targets = np.asarray(initial_positions, dtype=np.float32).copy()

while simulation_app.is_running():
    world.step(render=True)
    if not world.is_playing():
        continue

    current_slider_targets = np.array(
        [joint_models[name].get_value_as_float() for name in joint_names],
        dtype=np.float32,
    )
    target_position = np.array(
        [
            ik_models["x"].get_value_as_float(),
            ik_models["y"].get_value_as_float(),
            ik_models["z"].get_value_as_float(),
        ],
        dtype=np.float32,
    )
    rpy_offset = np.array(
        [
            ik_models["roll"].get_value_as_float(),
            ik_models["pitch"].get_value_as_float(),
            ik_models["yaw"].get_value_as_float(),
        ],
        dtype=np.float32,
    )
    target_orientation = quat_multiply_wxyz(
        euler_angles_to_quats(rpy_offset, degrees=True),
        ik_base_orientation,
    )
    lula_action, success = art_kinematics.compute_inverse_kinematics(
        target_position=target_position,
        target_orientation=target_orientation,
    )
    set_translate(ik_target_marker, target_position)
    ik_status.set_value("OK" if success else "FAIL")
    if success and lula_action is not None:
        full_targets = current_slider_targets.copy()
        lula_targets = np.asarray(lula_action.joint_positions, dtype=np.float32).reshape(-1)
        for lula_idx, joint_name in enumerate(lula_solver.get_joint_names()):
            full_targets[joint_name_to_index[joint_name]] = lula_targets[lula_idx]
        last_valid_arm_targets = full_targets.copy()
        robot.apply_action(ArticulationAction(joint_positions=full_targets))
    else:
        fallback_targets = last_valid_arm_targets.copy()
        fallback_targets[joint_name_to_index["joint7"]] = current_slider_targets[joint_name_to_index["joint7"]]
        fallback_targets[joint_name_to_index["joint8"]] = current_slider_targets[joint_name_to_index["joint8"]]
        robot.apply_action(ArticulationAction(joint_positions=fallback_targets))

simulation_app.close()
