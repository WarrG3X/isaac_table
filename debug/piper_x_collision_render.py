from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os

import omni.kit.commands
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ISAAC_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
DEFAULT_PIPER_USD = os.path.join(ISAAC_ROOT, "piper_isaac_sim", "USD", "piper_x_v1.usd")


parser = argparse.ArgumentParser()
parser.add_argument("--robot-usd", type=str, default=DEFAULT_PIPER_USD)
parser.add_argument("--robot-x", type=float, default=0.0)
parser.add_argument("--robot-y", type=float, default=0.12)
parser.add_argument("--robot-yaw", type=float, default=90.0)
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
        omni.usd.create_material_input(material_prim, "diffuse_color_constant", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f)
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
    candidate_tokens = {"environment", "groundplane", "ground_plane", "ground", "floor", "physicsscene", "camerasettings", "viewport_l", "viewport_r"}
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_prim_path + "/") and prim.GetName().lower() in candidate_tokens:
            prim.SetActive(False)


def set_visible(prim, visible):
    imageable = UsdGeom.Imageable(prim)
    if imageable:
        imageable.MakeVisible() if visible else imageable.MakeInvisible()


def prim_has_collision(prim):
    return prim.HasAPI(UsdPhysics.CollisionAPI) or prim.HasAPI(PhysxSchema.PhysxCollisionAPI)


def get_collision_approximation(prim):
    collision_api = PhysxSchema.PhysxCollisionAPI.Get(prim.GetStage(), prim.GetPath())
    if collision_api:
        attr = collision_api.GetApproximationAttr()
        if attr and attr.HasAuthoredValue():
            return attr.Get()
    return "unspecified"


def tint_collider_subtree(prim, material):
    if prim.IsA(UsdGeom.Gprim):
        bind_material(prim, material)
        UsdGeom.Gprim(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(0.92, 0.18, 0.18)])
        UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.default_)
        set_visible(prim, True)
    for child in prim.GetChildren():
        tint_collider_subtree(child, material)


def make_ancestor_chain_visible(prim, stop_path):
    current = prim
    while current and current.IsValid():
        if current.IsA(UsdGeom.Imageable):
            set_visible(current, True)
        if str(current.GetPath()) == stop_path:
            break
        current = current.GetParent()


def set_subtree_visible(prim, visible):
    if prim.IsA(UsdGeom.Imageable):
        set_visible(prim, visible)
    for child in prim.GetChildren():
        set_subtree_visible(child, visible)


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

bamboo_texture_path = os.path.join(REPO_ROOT, "materials", "nv_bamboo_desktop.jpg")
granite_texture_path = os.path.join(REPO_ROOT, "materials", "nv_granite_tile.jpg")
aluminum_anodized_dir = os.path.join(REPO_ROOT, "materials", "Aluminum_Anodized")
aluminum_anodized_basecolor = os.path.join(aluminum_anodized_dir, "Aluminum_Anodized_BaseColor.png")

floor_material_base = create_omnipbr_material(stage, "/World/Looks/FloorBase", color=(0.44, 0.46, 0.48), roughness=0.72)
floor_material = create_omnipbr_material(stage, "/World/Looks/Floor", color=(0.85, 0.85, 0.85), roughness=0.62, texture_path=granite_texture_path if os.path.exists(granite_texture_path) else None)
wood_material = create_omnipbr_material(stage, "/World/Looks/Wood", color=(0.72, 0.66, 0.54), roughness=0.38, texture_path=bamboo_texture_path if os.path.exists(bamboo_texture_path) else None)
wood_base_material = create_omnipbr_material(stage, "/World/Looks/WoodBase", color=(0.49, 0.35, 0.20), roughness=0.48)
leg_material = create_omnipbr_material(stage, "/World/Looks/Legs", color=(0.19, 0.12, 0.08), roughness=0.5)
bin_material = create_omnipbr_material(stage, "/World/Looks/Bin", color=(0.50, 0.50, 0.50), roughness=0.0, texture_path=aluminum_anodized_basecolor if os.path.exists(aluminum_anodized_basecolor) else None)
collider_material = create_omnipbr_material(stage, "/World/Looks/ColliderDebug", color=(0.92, 0.18, 0.18), roughness=0.12)

floor_base_prim = define_box(stage, "/World/Floor/Base", size=(6.8, 6.8, 0.02), translate=(0.0, 0.0, -0.01))
bind_material(floor_base_prim, floor_material_base)
floor_surface_prim = define_uv_plane(stage, "/World/Floor/Surface", size=(6.72, 6.72), translate=(0.0, 0.0, 0.001), uv_scale=(6.0, 6.0))
bind_material(floor_surface_prim, floor_material)

table_top_z = 0.75
table_top_thickness = 0.06
table_surface_z = table_top_z + table_top_thickness * 0.5
top_prim = define_box(stage, "/World/Table/Top", size=(1.4, 0.8, 0.06), translate=(0.0, 0.0, table_top_z))
bind_material(top_prim, wood_base_material)
table_surface_prim = define_uv_plane(stage, "/World/Table/TopSurface", size=(1.36, 0.76), translate=(0.0, 0.0, table_surface_z + 0.001))
bind_material(table_surface_prim, wood_material)

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

apply_static_collision([str(floor_base_prim.GetPath()), str(top_prim.GetPath()), *table_leg_paths])

mount_prim = UsdGeom.Xform.Define(stage, "/World/RobotMount/PiperX")
mount_xform = UsdGeom.Xformable(mount_prim.GetPrim())
mount_xform.AddTranslateOp().Set(Gf.Vec3d(args.robot_x, args.robot_y, table_surface_z))
mount_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, args.robot_yaw))
add_reference_to_stage(usd_path=os.path.abspath(args.robot_usd), prim_path="/World/RobotMount/PiperX/Robot")
deactivate_embedded_environment(stage, "/World/RobotMount/PiperX/Robot")

world.reset()

robot_root = stage.GetPrimAtPath("/World/RobotMount/PiperX/Robot")
collision_prims = []
approx_counts = {}
named_collision_prims = []
visual_prims = []

for prim in stage.Traverse():
    prim_path = str(prim.GetPath())
    if not prim_path.startswith("/World/RobotMount/PiperX/Robot/"):
        continue
    lower_path = prim_path.lower()
    if "/visuals/" in lower_path or lower_path.endswith("/visuals"):
        visual_prims.append(prim)
    if prim_has_collision(prim):
        collision_prims.append(prim)
        approx = get_collision_approximation(prim)
        approx_counts[approx] = approx_counts.get(approx, 0) + 1
    if "/collisions/" in lower_path or lower_path.endswith("/collisions") or "/collision/" in lower_path or lower_path.endswith("/collision"):
        named_collision_prims.append(prim)

if not collision_prims and named_collision_prims:
    collision_prims = named_collision_prims
    approx_counts = {"path_named_collision": len(collision_prims)}

for prim in visual_prims:
    set_subtree_visible(prim, False)

for prim in collision_prims:
    make_ancestor_chain_visible(prim, "/World/RobotMount/PiperX/Robot")
    tint_collider_subtree(prim, collider_material)
    set_visible(prim, True)

set_camera_view(
    eye=[1.35, -1.25, 1.2],
    target=[0.05, 0.18, 0.78],
    camera_prim_path="/OmniverseKit_Persp",
)

print(f"[CollisionRender] robot collider prims found: {len(collision_prims)}")
for approx, count in sorted(approx_counts.items()):
    print(f"[CollisionRender] approximation '{approx}': {count}")
for prim in collision_prims:
    print(f"[CollisionRender] {prim.GetPath()} approx={get_collision_approximation(prim)}")
    for child in prim.GetChildren():
        print(f"[CollisionRender]   child {child.GetPath()} type={child.GetTypeName()} instanceable={child.IsInstanceable()}")
        for grandchild in child.GetChildren():
            print(
                f"[CollisionRender]     grandchild {grandchild.GetPath()} "
                f"type={grandchild.GetTypeName()} instanceable={grandchild.IsInstanceable()}"
            )

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
