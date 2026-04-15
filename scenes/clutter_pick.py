import os

import numpy as np
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux

from .common import SceneInfo, add_robot, apply_static_collision, auto_generate_convex_colliders, bind_material, build_tabletop_with_cutouts, build_tray, compute_bbox_range, compute_bbox_volume, create_omnipbr_material, create_robot_camera, create_topdown_camera, define_box, define_uv_plane, list_ycb_assets, offset_prim_translate_z


ASSET_ROOT = r"C:\Users\Warra\Downloads\Assets\Isaac\5.1"
YCB_AXIS_ALIGNED_DIR = os.path.join(ASSET_ROOT, "Isaac", "Props", "YCB", "Axis_Aligned")


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


def default_scene_config(repo_root, robot_usd, ee_frame, robot_x, robot_y, robot_yaw, table_width, table_depth, tray_width, tray_depth, tray_height, tray_wall, num_objects, seed, max_volume):
    table_top_z = 0.75
    table_surface_z = table_top_z + 0.03
    return {
        "scene_schema_version": 1,
        "scene_name": "clutter_pick",
        "robot": {"usd_path": os.path.abspath(robot_usd), "x": float(robot_x), "y": float(robot_y), "yaw_deg": float(robot_yaw), "ee_frame": str(ee_frame)},
        "table": {"width": float(table_width), "depth": float(table_depth), "thickness": 0.06, "top_z": table_top_z, "leg_height": 0.72},
        "trays": {"width": float(tray_width), "depth": float(tray_depth), "height": float(tray_height), "wall": float(tray_wall), "source_center": [-0.38, 0.178], "target_center": [0.38, 0.178]},
        "objects": {"seed": int(seed), "max_volume": float(max_volume), "num_objects": int(num_objects), "asset_root": YCB_AXIS_ALIGNED_DIR, "spawned": []},
        "cameras": {"task_center": [0.0, 0.178, table_surface_z], "task_coverage_xy": [abs(0.38 - (-0.38)) + tray_width + 0.24, tray_depth + 0.28], "main_view_eye": [0.0, 2.15, 1.55], "main_view_target": [0.0, 0.20, 0.82]},
        "environment": {
            "name": "simple_room",
            "room_prim_path": "/World/Room",
            "deactivate_table_prim": "/World/Room/table_low_327",
            "floor_support_prim": "/World/Room/Towel_Room01_floor_bottom_218",
        },
        "task": {"name": "clutter_pick_source_to_target"},
        "repo_root": os.path.abspath(repo_root),
    }


def populate_spawned_objects(config):
    objects_cfg = config["objects"]
    if objects_cfg["spawned"]:
        return config
    tray_cfg = config["trays"]
    table_cfg = config["table"]
    table_surface_z = float(table_cfg["top_z"]) + 0.5 * float(table_cfg["thickness"])
    rng = np.random.default_rng(int(objects_cfg["seed"]))
    ycb_assets = [path for path in list_ycb_assets(objects_cfg["asset_root"]) if (compute_bbox_volume(path) or 0.0) < float(objects_cfg["max_volume"])]
    selected_count = min(int(objects_cfg["num_objects"]), len(ycb_assets))
    selected_indices = rng.choice(len(ycb_assets), size=selected_count, replace=False)
    inner_half_x = float(tray_cfg["width"]) * 0.5 - float(tray_cfg["wall"]) - 0.025
    inner_half_y = float(tray_cfg["depth"]) * 0.5 - float(tray_cfg["wall"]) - 0.025
    drop_base_z = table_surface_z + float(tray_cfg["height"]) + 0.08
    source_center = tray_cfg["source_center"]
    spawned = []
    for local_idx, asset_idx in enumerate(selected_indices):
        usd_path = ycb_assets[int(asset_idx)]
        x = float(source_center[0] + rng.uniform(-inner_half_x, inner_half_x))
        y = float(source_center[1] + rng.uniform(-inner_half_y, inner_half_y))
        z = float(drop_base_z + 0.06 * local_idx)
        yaw = float(rng.uniform(0.0, 360.0))
        quat = [float(np.cos(np.deg2rad(yaw) * 0.5)), 0.0, 0.0, float(np.sin(np.deg2rad(yaw) * 0.5))]
        bbox_range = compute_bbox_range(usd_path)
        bbox_size = bbox_range.GetSize() if bbox_range is not None else Gf.Vec3d(0.05, 0.05, 0.05)
        spawned.append({"name": os.path.basename(usd_path), "usd_path": usd_path, "prim_path": f"/World/Clutter/Object_{local_idx}", "initial_position": [x, y, z], "initial_orientation_wxyz": quat, "bbox_size": [float(bbox_size[0]), float(bbox_size[1]), float(bbox_size[2])]})
    objects_cfg["spawned"] = spawned
    objects_cfg["bank_size_after_filter"] = len(ycb_assets)
    return config


def build_scene(world, stage, config):
    config = populate_spawned_objects(config)
    repo_root = config["repo_root"]
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    for path in ["/World", "/World/Looks", "/World/Floor", "/World/Table", "/World/SourceTray", "/World/TargetTray", "/World/Clutter", "/World/Lights", "/World/RobotMount", "/World/Debug"]:
        UsdGeom.Xform.Define(stage, path)

    environment_cfg = config.get("environment", {})
    if environment_cfg.get("name") == "simple_room":
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            raise RuntimeError("Could not find Isaac Sim assets folder for Simple_Room")
        add_reference_to_stage(
            usd_path=assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd",
            prim_path=environment_cfg.get("room_prim_path", "/World/Room"),
        )
        room_table_path = environment_cfg.get("deactivate_table_prim")
        if room_table_path:
            room_table = stage.GetPrimAtPath(room_table_path)
            if room_table.IsValid():
                room_table.SetActive(False)

    bamboo_texture_path = os.path.join(repo_root, "materials", "nv_bamboo_desktop.jpg")
    granite_texture_path = os.path.join(repo_root, "materials", "nv_granite_tile.jpg")
    floor_material_base = create_omnipbr_material(stage, "/World/Looks/FloorBase", color=(0.44, 0.46, 0.48), roughness=0.72)
    floor_material = create_omnipbr_material(stage, "/World/Looks/Floor", color=(0.85, 0.85, 0.85), roughness=0.62, texture_path=granite_texture_path if os.path.exists(granite_texture_path) else None)
    wood_material = create_omnipbr_material(stage, "/World/Looks/Wood", color=(0.72, 0.66, 0.54), roughness=0.38, texture_path=bamboo_texture_path if os.path.exists(bamboo_texture_path) else None)
    wood_base_material = create_omnipbr_material(stage, "/World/Looks/WoodBase", color=(0.49, 0.35, 0.20), roughness=0.48)
    leg_material = create_omnipbr_material(stage, "/World/Looks/Legs", color=(0.19, 0.12, 0.08), roughness=0.5)
    source_tray_material = create_omnipbr_material(stage, "/World/Looks/SourceTray", color=(0.16, 0.24, 0.44), roughness=0.60, metallic=0.06)
    target_tray_material = create_omnipbr_material(stage, "/World/Looks/TargetTray", color=(0.46, 0.26, 0.16), roughness=0.62, metallic=0.04)

    floor_base_prim = define_box(stage, "/World/Floor/Base", size=(7.0, 7.0, 0.02), translate=(0.0, 0.0, -0.01))
    bind_material(floor_base_prim, floor_material_base)
    floor_surface_prim = define_uv_plane(stage, "/World/Floor/Surface", size=(6.92, 6.92), translate=(0.0, 0.0, 0.001), uv_scale=(6.0, 6.0))
    bind_material(floor_surface_prim, floor_material)

    table_cfg = config["table"]
    tray_cfg = config["trays"]
    cameras_cfg = config["cameras"]
    table_size = (float(table_cfg["width"]), float(table_cfg["depth"]), float(table_cfg["thickness"]))
    table_top_z = float(table_cfg["top_z"])
    leg_height = float(table_cfg["leg_height"])
    leg_dx = table_size[0] * 0.5 - 0.08
    leg_dy = table_size[1] * 0.5 - 0.08
    leg_center_z = leg_height * 0.5
    floor_support_prim = environment_cfg.get("floor_support_prim")
    if environment_cfg.get("name") == "simple_room" and floor_support_prim:
        floor_top_z = get_floor_top_z(stage, floor_support_prim)
        if floor_top_z is not None:
            leg_center_z = floor_top_z + leg_height * 0.5
            table_top_z = floor_top_z + leg_height + table_size[2] * 0.5
    table_surface_z = table_top_z + table_size[2] * 0.5
    table_leg_paths = []
    for name, position in {"LegFL": (leg_dx, leg_dy, leg_center_z), "LegFR": (leg_dx, -leg_dy, leg_center_z), "LegBL": (-leg_dx, leg_dy, leg_center_z), "LegBR": (-leg_dx, -leg_dy, leg_center_z)}.items():
        leg_prim = define_box(stage, f"/World/Table/{name}", size=(0.06, 0.06, leg_height), translate=position)
        bind_material(leg_prim, leg_material)
        table_leg_paths.append(str(leg_prim.GetPath()))

    source_tray_center = tuple(float(v) for v in tray_cfg["source_center"])
    target_tray_center = tuple(float(v) for v in tray_cfg["target_center"])
    source_tray_paths = build_tray(stage, "/World/SourceTray", center_xy=source_tray_center, surface_z=table_surface_z, outer_size=(float(tray_cfg["width"]), float(tray_cfg["depth"])), height=float(tray_cfg["height"]), wall=float(tray_cfg["wall"]), material=source_tray_material)
    target_tray_paths = build_tray(stage, "/World/TargetTray", center_xy=target_tray_center, surface_z=table_surface_z, outer_size=(float(tray_cfg["width"]), float(tray_cfg["depth"])), height=float(tray_cfg["height"]), wall=float(tray_cfg["wall"]), material=target_tray_material)
    for path in source_tray_paths + target_tray_paths:
        offset_prim_translate_z(stage, path, -float(tray_cfg["height"]))
    tabletop_paths = build_tabletop_with_cutouts(stage, "/World/Table", table_size=table_size, table_top_z=table_top_z, hole_specs=[(source_tray_center[0], source_tray_center[1], float(tray_cfg["width"]), float(tray_cfg["depth"])), (target_tray_center[0], target_tray_center[1], float(tray_cfg["width"]), float(tray_cfg["depth"]))], base_material=wood_base_material, surface_material=wood_material)

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

    apply_static_collision([str(floor_base_prim.GetPath()), *tabletop_paths, *table_leg_paths, *source_tray_paths, *target_tray_paths])
    robot_cfg = config["robot"]
    robot = add_robot(stage, robot_cfg["usd_path"], robot_cfg["x"], robot_cfg["y"], robot_cfg["yaw_deg"], table_surface_z)
    robot_camera_prim = create_robot_camera(stage, "/World/RobotMount/PiperX/Robot/piper_x_camera/camera_link/DebugCamera")
    task_center = (float(cameras_cfg["task_center"][0]), float(cameras_cfg["task_center"][1]), table_surface_z)
    topdown_camera_prim = create_topdown_camera(stage, "/World/Debug/TopDownCamera", center=task_center, coverage_xy=cameras_cfg["task_coverage_xy"])

    clutter_objects = []
    object_info = []
    for idx, entry in enumerate(config["objects"]["spawned"]):
        prim_path = entry.get("prim_path", f"/World/Clutter/Object_{idx}")
        add_reference_to_stage(usd_path=entry["usd_path"], prim_path=prim_path)
        auto_generate_convex_colliders(stage.GetPrimAtPath(prim_path))
        rigid = world.scene.add(SingleRigidPrim(prim_path=prim_path, name=f"clutter_object_{idx}", position=np.asarray(entry["initial_position"], dtype=np.float32), orientation=np.asarray(entry["initial_orientation_wxyz"], dtype=np.float32)))
        clutter_objects.append({"rigid": rigid, "name": entry["name"], "bbox_size": np.asarray(entry["bbox_size"], dtype=np.float32), "initial_position": np.asarray(entry["initial_position"], dtype=np.float32), "initial_orientation": np.asarray(entry["initial_orientation_wxyz"], dtype=np.float32)})
        object_info.append({"name": entry["name"], "usd_path": entry["usd_path"], "initial_position": list(entry["initial_position"]), "initial_orientation_wxyz": list(entry["initial_orientation_wxyz"]), "bbox_size": list(entry["bbox_size"])})

    set_camera_view(eye=cameras_cfg["main_view_eye"], target=cameras_cfg["main_view_target"], camera_prim_path="/OmniverseKit_Persp")
    if environment_cfg.get("name") == "simple_room":
        for prim_path in ["/World/Floor/Base", "/World/Floor/Surface", "/World/Lights/Dome", "/World/Lights/Sun"]:
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                prim.SetActive(False)
    return SceneInfo(scene_name="clutter_pick", robot=robot, robot_prim_path="/World/RobotMount/PiperX/Robot", robot_camera_prim_path=str(robot_camera_prim.GetPath()), topdown_camera_prim_path=str(topdown_camera_prim.GetPath()), table_surface_z=table_surface_z, task_center=task_center, task_coverage_xy=tuple(float(v) for v in cameras_cfg["task_coverage_xy"]), scene_config=config, extras={"source_tray_center": source_tray_center, "target_tray_center": target_tray_center, "clutter_objects": clutter_objects, "object_info": object_info, "tray_width": float(tray_cfg["width"]), "tray_depth": float(tray_cfg["depth"])})
