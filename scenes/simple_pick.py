import os

import numpy as np
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux

from .common import SceneInfo, add_robot, apply_static_collision, bind_material, create_omnipbr_material, create_robot_camera, create_topdown_camera, define_box, define_uv_plane


def default_scene_config(repo_root, robot_usd, ee_frame, robot_x, robot_y, robot_yaw, table_width, table_depth, cube_size):
    table_top_z = 0.75
    table_surface_z = table_top_z + 0.03
    source_pad_center = (-0.16, 0.22, table_surface_z + 0.002)
    target_pad_center = (0.16, 0.22, table_surface_z + 0.002)
    cube_start = [source_pad_center[0], source_pad_center[1], table_surface_z + cube_size * 0.5 + 0.002]
    return {
        "scene_schema_version": 1,
        "scene_name": "simple_pick",
        "robot": {"usd_path": os.path.abspath(robot_usd), "x": float(robot_x), "y": float(robot_y), "yaw_deg": float(robot_yaw), "ee_frame": str(ee_frame)},
        "table": {"width": float(table_width), "depth": float(table_depth), "thickness": 0.06, "top_z": table_top_z, "leg_height": 0.72},
        "pads": {"size_xy": [0.18, 0.18], "source_center": list(source_pad_center), "target_center": list(target_pad_center)},
        "cube": {"size": float(cube_size), "start_position": cube_start, "start_orientation_wxyz": [1.0, 0.0, 0.0, 0.0]},
        "cameras": {"task_center": [0.0, 0.22, table_surface_z], "task_coverage_xy": [0.68, 0.40], "main_view_eye": [0.0, 1.85, 1.55], "main_view_target": [0.0, 0.02, 0.82]},
        "task": {"name": "simple_pick_red_to_blue"},
        "repo_root": os.path.abspath(repo_root),
    }


def build_scene(world, stage, config):
    repo_root = config["repo_root"]
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for path in ["/World", "/World/Looks", "/World/Floor", "/World/Table", "/World/SourcePad", "/World/TargetPad", "/World/Lights", "/World/RobotMount", "/World/Debug"]:
        UsdGeom.Xform.Define(stage, path)

    bamboo_texture_path = os.path.join(repo_root, "materials", "nv_bamboo_desktop.jpg")
    granite_texture_path = os.path.join(repo_root, "materials", "nv_granite_tile.jpg")
    floor_material_base = create_omnipbr_material(stage, "/World/Looks/FloorBase", color=(0.44, 0.46, 0.48), roughness=0.72)
    floor_material = create_omnipbr_material(stage, "/World/Looks/Floor", color=(0.85, 0.85, 0.85), roughness=0.62, texture_path=granite_texture_path if os.path.exists(granite_texture_path) else None)
    wood_material = create_omnipbr_material(stage, "/World/Looks/Wood", color=(0.72, 0.66, 0.54), roughness=0.38, texture_path=bamboo_texture_path if os.path.exists(bamboo_texture_path) else None)
    wood_base_material = create_omnipbr_material(stage, "/World/Looks/WoodBase", color=(0.49, 0.35, 0.20), roughness=0.48)
    leg_material = create_omnipbr_material(stage, "/World/Looks/Legs", color=(0.19, 0.12, 0.08), roughness=0.5)
    source_pad_material = create_omnipbr_material(stage, "/World/Looks/SourcePad", color=(0.78, 0.16, 0.16), roughness=0.42)
    target_pad_material = create_omnipbr_material(stage, "/World/Looks/TargetPad", color=(0.16, 0.28, 0.82), roughness=0.42)

    floor_base_prim = define_box(stage, "/World/Floor/Base", size=(7.0, 7.0, 0.02), translate=(0.0, 0.0, -0.01))
    bind_material(floor_base_prim, floor_material_base)
    floor_surface_prim = define_uv_plane(stage, "/World/Floor/Surface", size=(6.92, 6.92), translate=(0.0, 0.0, 0.001), uv_scale=(6.0, 6.0))
    bind_material(floor_surface_prim, floor_material)

    table_cfg = config["table"]
    pads_cfg = config["pads"]
    cameras_cfg = config["cameras"]
    cube_cfg = config["cube"]
    table_size = (float(table_cfg["width"]), float(table_cfg["depth"]), float(table_cfg["thickness"]))
    table_top_z = float(table_cfg["top_z"])
    table_surface_z = table_top_z + table_size[2] * 0.5
    top_prim = define_box(stage, "/World/Table/Top", size=table_size, translate=(0.0, 0.0, table_top_z))
    bind_material(top_prim, wood_base_material)
    table_surface_prim = define_uv_plane(stage, "/World/Table/TopSurface", size=(table_size[0] - 0.04, table_size[1] - 0.04), translate=(0.0, 0.0, table_surface_z + 0.001))
    bind_material(table_surface_prim, wood_material)

    leg_height = float(table_cfg["leg_height"])
    leg_dx = table_size[0] * 0.5 - 0.08
    leg_dy = table_size[1] * 0.5 - 0.08
    table_leg_paths = []
    for name, position in {"LegFL": (leg_dx, leg_dy, leg_height * 0.5), "LegFR": (leg_dx, -leg_dy, leg_height * 0.5), "LegBL": (-leg_dx, leg_dy, leg_height * 0.5), "LegBR": (-leg_dx, -leg_dy, leg_height * 0.5)}.items():
        leg_prim = define_box(stage, f"/World/Table/{name}", size=(0.06, 0.06, leg_height), translate=position)
        bind_material(leg_prim, leg_material)
        table_leg_paths.append(str(leg_prim.GetPath()))

    pad_size = tuple(float(v) for v in pads_cfg["size_xy"])
    source_pad_center = tuple(float(v) for v in pads_cfg["source_center"])
    target_pad_center = tuple(float(v) for v in pads_cfg["target_center"])
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
    robot_cfg = config["robot"]
    robot = add_robot(stage, robot_cfg["usd_path"], robot_cfg["x"], robot_cfg["y"], robot_cfg["yaw_deg"], table_surface_z)
    robot_camera_prim = create_robot_camera(stage, "/World/RobotMount/PiperX/Robot/piper_x_camera/camera_link/DebugCamera")
    topdown_camera_prim = create_topdown_camera(stage, "/World/Debug/TopDownCamera", center=cameras_cfg["task_center"], coverage_xy=cameras_cfg["task_coverage_xy"])

    cube = DynamicCuboid(
        prim_path="/World/ObjectCube",
        name="simple_pick_cube",
        position=np.asarray(cube_cfg["start_position"], dtype=np.float32),
        scale=np.array([float(cube_cfg["size"])] * 3, dtype=np.float32),
        color=np.array([0.18, 0.76, 0.28], dtype=np.float32),
    )
    world.scene.add(cube)
    set_camera_view(eye=cameras_cfg["main_view_eye"], target=cameras_cfg["main_view_target"], camera_prim_path="/OmniverseKit_Persp")

    return SceneInfo(
        scene_name="simple_pick",
        robot=robot,
        robot_prim_path="/World/RobotMount/PiperX/Robot",
        robot_camera_prim_path=str(robot_camera_prim.GetPath()),
        topdown_camera_prim_path=str(topdown_camera_prim.GetPath()),
        table_surface_z=table_surface_z,
        task_center=tuple(float(v) for v in cameras_cfg["task_center"]),
        task_coverage_xy=tuple(float(v) for v in cameras_cfg["task_coverage_xy"]),
        scene_config=config,
        extras={"source_pad_center": source_pad_center, "target_pad_center": target_pad_center, "pad_size": pad_size, "cube": cube, "initial_cube_position": np.asarray(cube_cfg["start_position"], dtype=np.float32), "initial_cube_orientation": np.asarray(cube_cfg["start_orientation_wxyz"], dtype=np.float32)},
    )
