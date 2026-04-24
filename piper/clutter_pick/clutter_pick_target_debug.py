from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os
import sys
import threading

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ISAAC_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scenes.common import (
    apply_static_collision,
    auto_generate_convex_colliders,
    bind_material,
    build_tabletop_with_cutouts,
    build_tray,
    compute_bbox_range,
    compute_bbox_volume,
    create_omnipbr_material,
    define_box,
    define_uv_plane,
    list_ycb_assets,
    offset_prim_translate_z,
)
from scenes.clutter_pick import YCB_AXIS_ALIGNED_DIR, default_scene_config, get_floor_top_z


DEFAULT_TABLE_WIDTH = 1.90
DEFAULT_TABLE_DEPTH = 1.55
DEFAULT_TRAY_WIDTH = 0.60
DEFAULT_TRAY_DEPTH = 0.42
DEFAULT_TRAY_HEIGHT = 0.1275
DEFAULT_DEBUG_TRAY_HEIGHT = 0.035
DEFAULT_TRAY_WALL = 0.012
DEFAULT_NUM_OBJECTS = 12
DEFAULT_SEED = 42
DEFAULT_MAX_VOLUME = 0.002
DEFAULT_ROBOT_X = 0.0
DEFAULT_ROBOT_Y = -0.05
DEFAULT_ROBOT_YAW = 0.0
DEFAULT_EE_FRAME = "link6"
DEFAULT_PIPER_USD = os.path.join(ISAAC_ROOT, "piper_isaac_sim", "USD", "piper_x_v1.usd")
TARGET_USD_NAME = "061_foam_brick.usd"
TARGET_CORNER_INSET_X_FRACTION = 0.20
TARGET_CORNER_INSET_Y_FRACTION = 0.20
TARGET_DROP_HEIGHT_OFFSET_Z = 0.02
TARGET_KEEP_OUT_HALF_EXTENT_X = 0.10
TARGET_KEEP_OUT_HALF_EXTENT_Y = 0.08


parser = argparse.ArgumentParser()
parser.add_argument("--table-width", type=float, default=DEFAULT_TABLE_WIDTH)
parser.add_argument("--table-depth", type=float, default=DEFAULT_TABLE_DEPTH)
parser.add_argument("--tray-width", type=float, default=DEFAULT_TRAY_WIDTH)
parser.add_argument("--tray-depth", type=float, default=DEFAULT_TRAY_DEPTH)
parser.add_argument("--tray-height", type=float, default=DEFAULT_DEBUG_TRAY_HEIGHT)
parser.add_argument("--tray-wall", type=float, default=DEFAULT_TRAY_WALL)
parser.add_argument("--num-objects", type=int, default=DEFAULT_NUM_OBJECTS)
parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
parser.add_argument("--max-volume", type=float, default=DEFAULT_MAX_VOLUME)
args, _ = parser.parse_known_args()


def build_scene_config(seed: int) -> dict:
    config = default_scene_config(
        repo_root=REPO_ROOT,
        robot_usd=DEFAULT_PIPER_USD,
        ee_frame=DEFAULT_EE_FRAME,
        robot_x=DEFAULT_ROBOT_X,
        robot_y=DEFAULT_ROBOT_Y,
        robot_yaw=DEFAULT_ROBOT_YAW,
        table_width=args.table_width,
        table_depth=args.table_depth,
        tray_width=args.tray_width,
        tray_depth=args.tray_depth,
        tray_height=args.tray_height,
        tray_wall=args.tray_wall,
        num_objects=args.num_objects,
        seed=seed,
        max_volume=args.max_volume,
    )
    config["scene_name"] = "temp_clutter_reset_debug"
    config["task"]["name"] = "temp_clutter_reset_debug"
    return config


def generate_spawned_entries(seed: int, scene_config: dict, table_surface_z: float, object_specs: list[dict] | None = None):
    source_center = tuple(float(v) for v in scene_config["trays"]["source_center"])
    tray_width = float(scene_config["trays"]["width"])
    tray_depth = float(scene_config["trays"]["depth"])
    tray_wall = float(scene_config["trays"]["wall"])
    drop_base_z = table_surface_z + float(scene_config["trays"]["height"]) + 0.08
    inner_half_x = tray_width * 0.5 - tray_wall - 0.025
    inner_half_y = tray_depth * 0.5 - tray_wall - 0.025
    rng = np.random.default_rng(seed)

    if object_specs is None:
        ycb_assets = [path for path in list_ycb_assets(YCB_AXIS_ALIGNED_DIR) if (compute_bbox_volume(path) or 0.0) < args.max_volume]
        if not ycb_assets:
            raise RuntimeError(f"No YCB assets remain after applying max-volume filter {args.max_volume}")
        target_usd_path = os.path.abspath(os.path.join(YCB_AXIS_ALIGNED_DIR, TARGET_USD_NAME))
        random_asset_pool = [path for path in ycb_assets if os.path.abspath(path) != target_usd_path]
        selected_count = min(args.num_objects, len(random_asset_pool))
        selected_indices = rng.choice(len(random_asset_pool), size=selected_count, replace=False)
        base_specs = []
        for asset_idx in selected_indices:
            usd_path = random_asset_pool[int(asset_idx)]
            bbox_range = compute_bbox_range(usd_path)
            bbox_size = bbox_range.GetSize() if bbox_range is not None else Gf.Vec3d(0.05, 0.05, 0.05)
            base_specs.append(
                {
                    "name": os.path.basename(usd_path),
                    "usd_path": usd_path,
                    "bbox_size": [float(bbox_size[0]), float(bbox_size[1]), float(bbox_size[2])],
                    "is_target": False,
                }
            )
        target_bbox_range = compute_bbox_range(target_usd_path)
        target_bbox_size = target_bbox_range.GetSize() if target_bbox_range is not None else Gf.Vec3d(0.05, 0.05, 0.05)
        base_specs.append(
            {
                "name": os.path.basename(target_usd_path),
                "usd_path": target_usd_path,
                "bbox_size": [float(target_bbox_size[0]), float(target_bbox_size[1]), float(target_bbox_size[2])],
                "is_target": True,
            }
        )
        scene_config["objects"]["bank_size_after_filter"] = len(ycb_assets)
    else:
        base_specs = object_specs

    spawned = []
    random_specs = [spec for spec in base_specs if not bool(spec.get("is_target", False))]
    target_specs = [spec for spec in base_specs if bool(spec.get("is_target", False))]
    target_x = float(source_center[0] + inner_half_x - inner_half_x * TARGET_CORNER_INSET_X_FRACTION)
    target_y = float(source_center[1] - inner_half_y + inner_half_y * TARGET_CORNER_INSET_Y_FRACTION)

    for local_idx, spec in enumerate(random_specs):
        for _ in range(128):
            candidate_x = float(source_center[0] + rng.uniform(-inner_half_x, inner_half_x))
            candidate_y = float(source_center[1] + rng.uniform(-inner_half_y, inner_half_y))
            inside_keepout = (
                abs(candidate_x - target_x) <= TARGET_KEEP_OUT_HALF_EXTENT_X
                and abs(candidate_y - target_y) <= TARGET_KEEP_OUT_HALF_EXTENT_Y
            )
            if not inside_keepout:
                x = candidate_x
                y = candidate_y
                break
        else:
            x = float(source_center[0] - inner_half_x + 0.02)
            y = float(source_center[1] + inner_half_y - 0.02)
        z = float(drop_base_z + 0.06 * local_idx)
        yaw = float(rng.uniform(0.0, 360.0))
        quat = [float(np.cos(np.deg2rad(yaw) * 0.5)), 0.0, 0.0, float(np.sin(np.deg2rad(yaw) * 0.5))]
        spawned.append(
            {
                **spec,
                "prim_path": spec.get("prim_path", f"/World/Clutter/Object_{local_idx}"),
                "initial_position": [x, y, z],
                "initial_orientation_wxyz": quat,
            }
        )
    for spec in target_specs:
        local_idx = len(spawned)
        x = target_x
        y = target_y
        z = float(drop_base_z + TARGET_DROP_HEIGHT_OFFSET_Z)
        yaw = float(rng.uniform(0.0, 360.0))
        quat = [float(np.cos(np.deg2rad(yaw) * 0.5)), 0.0, 0.0, float(np.sin(np.deg2rad(yaw) * 0.5))]
        spawned.append(
            {
                **spec,
                "prim_path": spec.get("prim_path", f"/World/Clutter/Object_{local_idx}"),
                "initial_position": [x, y, z],
                "initial_orientation_wxyz": quat,
            }
        )
    return spawned


current_seed = args.seed
reset_requested = False


def build_world_for_seed(seed: int):
    local_world = World(stage_units_in_meters=1.0)
    local_stage = local_world.stage
    UsdGeom.SetStageUpAxis(local_stage, UsdGeom.Tokens.z)
    for path in ["/World", "/World/Looks", "/World/Floor", "/World/Table", "/World/SourceTray", "/World/TargetTray", "/World/Clutter", "/World/Lights"]:
        UsdGeom.Xform.Define(local_stage, path)

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not find Isaac Sim assets folder for Simple_Room")
    add_reference_to_stage(
        usd_path=assets_root_path + "/Isaac/Environments/Simple_Room/simple_room.usd",
        prim_path="/World/Room",
    )
    room_table = local_stage.GetPrimAtPath("/World/Room/table_low_327")
    if room_table.IsValid():
        room_table.SetActive(False)

    bamboo_texture_path = os.path.join(REPO_ROOT, "materials", "nv_bamboo_desktop.jpg")
    granite_texture_path = os.path.join(REPO_ROOT, "materials", "nv_granite_tile.jpg")
    floor_material_base = create_omnipbr_material(local_stage, "/World/Looks/FloorBase", color=(0.44, 0.46, 0.48), roughness=0.72)
    floor_material = create_omnipbr_material(local_stage, "/World/Looks/Floor", color=(0.85, 0.85, 0.85), roughness=0.62, texture_path=granite_texture_path if os.path.exists(granite_texture_path) else None)
    wood_material = create_omnipbr_material(local_stage, "/World/Looks/Wood", color=(0.72, 0.66, 0.54), roughness=0.38, texture_path=bamboo_texture_path if os.path.exists(bamboo_texture_path) else None)
    wood_base_material = create_omnipbr_material(local_stage, "/World/Looks/WoodBase", color=(0.49, 0.35, 0.20), roughness=0.48)
    leg_material = create_omnipbr_material(local_stage, "/World/Looks/Legs", color=(0.19, 0.12, 0.08), roughness=0.5)
    source_tray_material = create_omnipbr_material(local_stage, "/World/Looks/SourceTray", color=(0.16, 0.24, 0.44), roughness=0.60, metallic=0.06)
    target_tray_material = create_omnipbr_material(local_stage, "/World/Looks/TargetTray", color=(0.46, 0.26, 0.16), roughness=0.62, metallic=0.04)

    floor_prim = define_box(local_stage, "/World/Floor/Base", size=(7.0, 7.0, 0.02), translate=(0.0, 0.0, -0.01))
    bind_material(floor_prim, floor_material_base)
    floor_surface = define_uv_plane(local_stage, "/World/Floor/Surface", size=(6.92, 6.92), translate=(0.0, 0.0, 0.001), uv_scale=(6.0, 6.0))
    bind_material(floor_surface, floor_material)

    scene_config = build_scene_config(seed)
    table_cfg = scene_config["table"]
    tray_cfg = scene_config["trays"]
    table_size = (float(table_cfg["width"]), float(table_cfg["depth"]), float(table_cfg["thickness"]))
    table_top_z = float(table_cfg["top_z"])
    leg_height = float(table_cfg["leg_height"])
    leg_dx = table_size[0] * 0.5 - 0.08
    leg_dy = table_size[1] * 0.5 - 0.08
    leg_center_z = leg_height * 0.5
    floor_top_z = get_floor_top_z(local_stage, scene_config["environment"]["floor_support_prim"])
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
        leg_prim = define_box(local_stage, f"/World/Table/{name}", size=(0.06, 0.06, leg_height), translate=position)
        bind_material(leg_prim, leg_material)
        table_collision_paths.append(str(leg_prim.GetPath()))

    source_tray_center = tuple(float(v) for v in tray_cfg["source_center"])
    target_tray_center = tuple(float(v) for v in tray_cfg["target_center"])
    source_tray_paths = build_tray(local_stage, "/World/SourceTray", center_xy=source_tray_center, surface_z=table_surface_z, outer_size=(float(tray_cfg["width"]), float(tray_cfg["depth"])), height=float(tray_cfg["height"]), wall=float(tray_cfg["wall"]), material=source_tray_material)
    target_tray_paths = build_tray(local_stage, "/World/TargetTray", center_xy=target_tray_center, surface_z=table_surface_z, outer_size=(float(tray_cfg["width"]), float(tray_cfg["depth"])), height=float(tray_cfg["height"]), wall=float(tray_cfg["wall"]), material=target_tray_material)
    for path in source_tray_paths + target_tray_paths:
        offset_prim_translate_z(local_stage, path, -float(tray_cfg["height"]))
    tabletop_paths = build_tabletop_with_cutouts(
        local_stage,
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

    dome_light = UsdLux.DomeLight.Define(local_stage, Sdf.Path("/World/Lights/Dome"))
    dome_light.CreateIntensityAttr(950.0)
    dome_light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
    sun_light = UsdLux.DistantLight.Define(local_stage, Sdf.Path("/World/Lights/Sun"))
    sun_light.CreateIntensityAttr(3200.0)
    sun_light.CreateAngleAttr(0.6)
    sun_light.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
    UsdGeom.Xformable(sun_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))
    table_fill = UsdLux.RectLight.Define(local_stage, Sdf.Path("/World/Lights/TableFill"))
    table_fill.CreateIntensityAttr(2000.0)
    table_fill.CreateWidthAttr(2.4)
    table_fill.CreateHeightAttr(1.2)
    table_fill.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    UsdGeom.Xformable(table_fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3f(0.3, 0.05, 1.95))
    UsdGeom.Xformable(table_fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))

    set_camera_view(
        eye=[0.0, 1.90, 1.45],
        target=[0.0, 0.20, 0.80],
        camera_prim_path="/OmniverseKit_Persp",
    )
    for prim_path in ["/World/Floor/Base", "/World/Floor/Surface", "/World/Lights/Dome", "/World/Lights/Sun"]:
        prim = local_stage.GetPrimAtPath(prim_path)
        if prim.IsValid():
            prim.SetActive(False)

    scene_config["objects"]["spawned"] = generate_spawned_entries(seed, scene_config, table_surface_z)
    clutter_entries = []
    for idx, entry in enumerate(scene_config["objects"]["spawned"]):
        prim_path = entry.get("prim_path", f"/World/Clutter/Object_{idx}")
        add_reference_to_stage(usd_path=entry["usd_path"], prim_path=prim_path)
        auto_generate_convex_colliders(local_stage.GetPrimAtPath(prim_path))
        rigid = local_world.scene.add(
            SingleRigidPrim(
                prim_path=prim_path,
                name=f"clutter_object_{idx}",
                position=np.asarray(entry["initial_position"], dtype=np.float32),
                orientation=np.asarray(entry["initial_orientation_wxyz"], dtype=np.float32),
            )
        )
        clutter_entries.append(
            {
                "rigid": rigid,
                "name": entry["name"],
                "usd_path": entry["usd_path"],
                "bbox_size": entry["bbox_size"],
                "is_target": bool(entry.get("is_target", False)),
                "initial_position": np.asarray(entry["initial_position"], dtype=np.float32),
                "initial_orientation": np.asarray(entry["initial_orientation_wxyz"], dtype=np.float32),
            }
        )

    local_world.reset()
    for entry in clutter_entries:
        entry["rigid"].initialize()
    print(f"[TempClutterReset] seed={seed} objects={len(clutter_entries)}")
    for entry in scene_config["objects"]["spawned"]:
        prefix = "[target]" if entry.get("is_target", False) else "        "
        print(f"{prefix} {entry['name']} @ {tuple(round(v, 4) for v in entry['initial_position'])}")
    print("[TempClutterReset] Press Enter to respawn clutter with seed+1...")
    return local_world, clutter_entries, table_surface_z


def reset_clutter_for_seed(seed: int, clutter_entries: list[dict], table_surface_z: float):
    object_specs = [
        {
            "name": entry["name"],
            "usd_path": entry["usd_path"],
            "bbox_size": entry["bbox_size"],
            "is_target": entry["is_target"],
        }
        for entry in clutter_entries
    ]
    scene_config = build_scene_config(seed)
    spawned_entries = generate_spawned_entries(seed, scene_config, table_surface_z, object_specs=object_specs)
    scene_config["objects"]["spawned"] = spawned_entries
    for entry, spawn in zip(clutter_entries, spawned_entries):
        position = np.asarray(spawn["initial_position"], dtype=np.float32)
        orientation = np.asarray(spawn["initial_orientation_wxyz"], dtype=np.float32)
        entry["initial_position"] = position
        entry["initial_orientation"] = orientation
        entry["rigid"].set_world_pose(position=position, orientation=orientation)
        entry["rigid"].set_linear_velocity(np.zeros(3, dtype=np.float32))
        entry["rigid"].set_angular_velocity(np.zeros(3, dtype=np.float32))
    print(f"[TempClutterReset] seed={seed} objects={len(clutter_entries)}")
    for entry in scene_config["objects"]["spawned"]:
        prefix = "[target]" if entry.get("is_target", False) else "        "
        print(f"{prefix} {entry['name']} @ {tuple(round(v, 4) for v in entry['initial_position'])}")
    print("[TempClutterReset] Press Enter to respawn clutter with seed+1...")


def stdin_loop():
    global reset_requested
    while True:
        try:
            input()
        except EOFError:
            return
        reset_requested = True


threading.Thread(target=stdin_loop, daemon=True).start()
world, clutter_entries, table_surface_z = build_world_for_seed(current_seed)

while simulation_app.is_running():
    world.step(render=True)
    if reset_requested:
        reset_requested = False
        current_seed += 1
        reset_clutter_for_seed(current_seed, clutter_entries, table_surface_z)

simulation_app.close()
