from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os

import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import omni.kit.commands
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, DynamicSphere
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from PIL import Image
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object-source",
    choices=["prims", "ycb"],
    default="prims",
    help="Object source to populate into the bin. YCB is a placeholder until Isaac assets are available.",
)
parser.add_argument("--num-objects", type=int, default=6, help="Number of clutter objects to drop into the bin.")
parser.add_argument("--seed", type=int, default=7, help="Deterministic seed for object placement.")
parser.add_argument(
    "--save-camera-debug",
    action="store_true",
    help="Save RGB and depth snapshots after objects settle in the bin.",
)
args, _ = parser.parse_known_args()


def define_box(stage, prim_path, size, translate):
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*translate))

    cube = UsdGeom.Cube.Define(stage, f"{prim_path}/Geom")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*size))
    return cube.GetPrim()


def create_omnipbr_material(stage, material_path, color=None, roughness=0.5, metallic=0.0, texture_path=None):
    mtl_created_list = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniPBR.mdl",
        mtl_name="OmniPBR",
        mtl_created_list=mtl_created_list,
    )

    created_path = mtl_created_list[0]
    material_prim = stage.GetPrimAtPath(created_path)

    if created_path != material_path:
        omni.kit.commands.execute("MovePrim", path_from=created_path, path_to=material_path)
        material_prim = stage.GetPrimAtPath(material_path)

    if color is not None:
        omni.usd.create_material_input(
            material_prim, "diffuse_color_constant", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f
        )
    omni.usd.create_material_input(material_prim, "reflection_roughness_constant", roughness, Sdf.ValueTypeNames.Float)
    omni.usd.create_material_input(material_prim, "metallic_constant", metallic, Sdf.ValueTypeNames.Float)

    if texture_path:
        omni.usd.create_material_input(material_prim, "diffuse_texture", texture_path, Sdf.ValueTypeNames.Asset)

    return UsdShade.Material(material_prim)


def create_omniglass_material(stage, material_path, color, ior=1.25):
    mtl_created_list = []
    omni.kit.commands.execute(
        "CreateAndBindMdlMaterialFromLibrary",
        mdl_name="OmniGlass.mdl",
        mtl_name="OmniGlass",
        mtl_created_list=mtl_created_list,
    )

    created_path = mtl_created_list[0]
    material_prim = stage.GetPrimAtPath(created_path)

    if created_path != material_path:
        omni.kit.commands.execute("MovePrim", path_from=created_path, path_to=material_path)
        material_prim = stage.GetPrimAtPath(material_path)

    omni.usd.create_material_input(material_prim, "glass_color", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f)
    omni.usd.create_material_input(material_prim, "glass_ior", ior, Sdf.ValueTypeNames.Float)

    return UsdShade.Material(material_prim)


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def tile_name(ix, iy):
    x = f"n{abs(ix)}" if ix < 0 else f"p{ix}"
    y = f"n{abs(iy)}" if iy < 0 else f"p{iy}"
    return f"Tile_{x}_{y}"


def apply_static_collision(prim_paths):
    for prim_path in prim_paths:
        GeometryPrim(prim_path).apply_collision_apis()


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
        spawned.append(obj)

    return spawned


def spawn_objects(world, rng, source, count, bin_center_y, table_surface_z):
    UsdGeom.Xform.Define(stage, "/World/Clutter")

    if source == "prims":
        return spawn_primitive_objects(world, rng, count, bin_center_y, table_surface_z)

    raise RuntimeError(
        "YCB spawning is not available in this install yet because the Isaac asset pack is missing. "
        "Use --object-source prims for now, then switch to ycb once /Isaac/Props/YCB assets are installed."
    )


def objects_are_settled(objects, linear_threshold=0.03, angular_threshold=0.6):
    for obj in objects:
        linear = np.linalg.norm(np.asarray(obj.get_linear_velocity()))
        angular = np.linalg.norm(np.asarray(obj.get_angular_velocity()))
        if linear > linear_threshold or angular > angular_threshold:
            return False
    return True


world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()
rng = np.random.default_rng(args.seed)

UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Floor")
UsdGeom.Xform.Define(stage, "/World/Table")
UsdGeom.Xform.Define(stage, "/World/Bin")
UsdGeom.Xform.Define(stage, "/World/Lights")

try:
    assets_root = get_assets_root_path()
except Exception:
    assets_root = None
marble_texture_path = None
if assets_root:
    candidate = assets_root + "/Isaac/Samples/DR/Materials/Textures/marble_tile.png"
    if os.path.exists(candidate):
        marble_texture_path = candidate

floor_material_dark = create_omnipbr_material(
    stage,
    "/World/Looks/FloorDark",
    color=(0.33, 0.34, 0.36),
    roughness=0.7,
    texture_path=marble_texture_path,
)
floor_material_light = create_omnipbr_material(
    stage,
    "/World/Looks/FloorLight",
    color=(0.60, 0.61, 0.62),
    roughness=0.55,
    texture_path=marble_texture_path,
)
wood_material = create_omnipbr_material(
    stage,
    "/World/Looks/Wood",
    color=(0.54, 0.39, 0.24),
    roughness=0.45,
)
leg_material = create_omnipbr_material(
    stage,
    "/World/Looks/Legs",
    color=(0.19, 0.12, 0.08),
    roughness=0.5,
)
bin_material = create_omniglass_material(
    stage,
    "/World/Looks/Bin",
    color=(0.45, 0.62, 0.88),
    ior=1.18,
)

tile_size = 0.75
tile_thickness = 0.02
half_span = 4
floor_geom_paths = []
for ix in range(-half_span, half_span + 1):
    for iy in range(-half_span, half_span + 1):
        tile_prim = define_box(
            stage,
            f"/World/Floor/{tile_name(ix, iy)}",
            size=(tile_size, tile_size, tile_thickness),
            translate=(ix * tile_size, iy * tile_size, -tile_thickness * 0.5),
        )
        bind_material(tile_prim, floor_material_dark if (ix + iy) % 2 == 0 else floor_material_light)
        floor_geom_paths.append(str(tile_prim.GetPath()))

top_prim = define_box(
    stage,
    "/World/Table/Top",
    size=(1.4, 0.8, 0.06),
    translate=(0.0, 0.0, 0.75),
)
bind_material(top_prim, wood_material)

leg_height = 0.72
leg_size = (0.06, 0.06, leg_height)
leg_center_z = leg_height * 0.5
table_leg_paths = []

for name, position in {
    "LegFL": (0.62, 0.32, leg_center_z),
    "LegFR": (0.62, -0.32, leg_center_z),
    "LegBL": (-0.62, 0.32, leg_center_z),
    "LegBR": (-0.62, -0.32, leg_center_z),
}.items():
    leg_prim = define_box(stage, f"/World/Table/{name}", size=leg_size, translate=position)
    bind_material(leg_prim, leg_material)
    table_leg_paths.append(str(leg_prim.GetPath()))

table_top_z = 0.75
table_top_thickness = 0.06
table_surface_z = table_top_z + table_top_thickness * 0.5
table_half_depth = 0.8 * 0.5

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
UsdGeom.Xformable(table_fill.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 2.6))
UsdGeom.Xformable(table_fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(0.0, 180.0, 0.0))

set_camera_view(
    eye=[3.2, -3.0, 2.2],
    target=[0.0, 0.0, 0.72],
    camera_prim_path="/OmniverseKit_Persp",
)

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

world.reset()
topdown_camera.initialize()
topdown_camera.attach_annotator("distance_to_image_plane")

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

apply_static_collision(floor_geom_paths + [str(top_prim.GetPath())] + table_leg_paths + bin_geom_paths)

clutter_objects = spawn_objects(world, rng, args.object_source, args.num_objects, bin_center_y, table_surface_z)

settled = False
settle_counter = 0
settle_counter_target = 30
capture_done = False

for _ in range(15):
    world.step(render=True)

while simulation_app.is_running():
    world.step(render=True)

    if not settled:
        if objects_are_settled(clutter_objects):
            settle_counter += 1
        else:
            settle_counter = 0
        settled = settle_counter >= settle_counter_target

    if settled and not capture_done:
        save_camera_debug(topdown_camera, os.path.join(os.getcwd(), "camera_debug"))
        capture_done = True

        if args.save_camera_debug:
            print("[TopDownCamera] capture complete after clutter settled")

simulation_app.close()
