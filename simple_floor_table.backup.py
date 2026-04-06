from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os

import omni.kit.commands
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade


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


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def tile_name(ix, iy):
    x = f"n{abs(ix)}" if ix < 0 else f"p{ix}"
    y = f"n{abs(iy)}" if iy < 0 else f"p{iy}"
    return f"Tile_{x}_{y}"


world = World(stage_units_in_meters=1.0)
stage = omni.usd.get_context().get_stage()

UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Floor")
UsdGeom.Xform.Define(stage, "/World/Table")
UsdGeom.Xform.Define(stage, "/World/Lights")

assets_root = get_assets_root_path()
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

# Procedural tiled floor for visual texture.
tile_size = 0.75
tile_thickness = 0.02
half_span = 4
for ix in range(-half_span, half_span + 1):
    for iy in range(-half_span, half_span + 1):
        tile_prim = define_box(
            stage,
            f"/World/Floor/{tile_name(ix, iy)}",
            size=(tile_size, tile_size, tile_thickness),
            translate=(ix * tile_size, iy * tile_size, -tile_thickness * 0.5),
        )
        bind_material(tile_prim, floor_material_dark if (ix + iy) % 2 == 0 else floor_material_light)

# Tabletop centered at the world origin.
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

for name, position in {
    "LegFL": (0.62, 0.32, leg_center_z),
    "LegFR": (0.62, -0.32, leg_center_z),
    "LegBL": (-0.62, 0.32, leg_center_z),
    "LegBR": (-0.62, -0.32, leg_center_z),
}.items():
    leg_prim = define_box(stage, f"/World/Table/{name}", size=leg_size, translate=position)
    bind_material(leg_prim, leg_material)

# Lighting rig: soft ambient dome + directional key + overhead area light.
dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
dome_light.CreateIntensityAttr(900.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
UsdGeom.Xformable(dome_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

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

world.reset()

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
