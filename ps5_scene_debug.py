from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import os

import numpy as np
import omni.kit.commands
import pygame
from isaacsim.core.api import World
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


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
    if texture_path:
        omni.usd.create_material_input(material_prim, "diffuse_texture", texture_path, Sdf.ValueTypeNames.Asset)
    return UsdShade.Material(material_prim)


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def apply_static_collision(prim_paths):
    for prim_path in prim_paths:
        GeometryPrim(prim_path).apply_collision_apis()


def set_translate(prim, translate):
    tx, ty, tz = float(translate[0]), float(translate[1]), float(translate[2])
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(tx, ty, tz))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))


def create_marker(stage, prim_path, radius=0.03):
    marker = UsdGeom.Sphere.Define(stage, prim_path)
    marker.CreateRadiusAttr(radius)
    return marker.GetPrim()


def init_controller(index):
    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    print(f"[PS5SceneDebug] pygame joysticks={count}")
    if count == 0:
        raise RuntimeError("No controller detected by pygame")
    if index >= count:
        raise RuntimeError(f"Controller index {index} out of range")
    joystick = pygame.joystick.Joystick(index)
    joystick.init()
    print(
        f"[PS5SceneDebug] name={joystick.get_name()} "
        f"axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}"
    )
    return joystick


parser = argparse.ArgumentParser()
parser.add_argument("--controller", type=int, default=0)
parser.add_argument("--translation-scale", type=float, default=0.01, help="Meters per frame at full stick/trigger.")
parser.add_argument("--deadband", type=float, default=0.10)
args, _ = parser.parse_known_args()

world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Floor")
UsdGeom.Xform.Define(stage, "/World/Table")
UsdGeom.Xform.Define(stage, "/World/Debug")
UsdGeom.Xform.Define(stage, "/World/Lights")

granite_texture_path = os.path.join(SCRIPT_DIR, "materials", "nv_granite_tile.jpg")
bamboo_texture_path = os.path.join(SCRIPT_DIR, "materials", "nv_bamboo_desktop.jpg")

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
marker_material = create_omnipbr_material(stage, "/World/Looks/Marker", color=(0.82, 0.16, 0.12), roughness=0.25)

floor_prim = define_box(stage, "/World/Floor/Base", size=(6.8, 6.8, 0.02), translate=(0.0, 0.0, -0.01))
bind_material(floor_prim, floor_material_base)
floor_surface = define_uv_plane(stage, "/World/Floor/Surface", size=(6.72, 6.72), translate=(0.0, 0.0, 0.001), uv_scale=(6.0, 6.0))
bind_material(floor_surface, floor_material)

table_top_z = 0.75
table_surface_z = table_top_z + 0.03
top_prim = define_box(stage, "/World/Table/Top", size=(1.4, 0.8, 0.06), translate=(0.0, 0.0, table_top_z))
bind_material(top_prim, wood_base_material)
table_surface = define_uv_plane(stage, "/World/Table/TopSurface", size=(1.36, 0.76), translate=(0.0, 0.0, table_surface_z + 0.001))
bind_material(table_surface, wood_material)

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

apply_static_collision([str(floor_prim.GetPath()), str(top_prim.GetPath()), *table_leg_paths])

marker_prim = create_marker(stage, "/World/Debug/Marker")
bind_material(marker_prim, marker_material)
marker_position = np.array([0.0, 0.10, table_surface_z + 0.08], dtype=np.float32)
set_translate(marker_prim, marker_position)

dome_light = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
dome_light.CreateIntensityAttr(950.0)
dome_light.CreateColorAttr(Gf.Vec3f(0.92, 0.95, 1.0))
sun_light = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Sun"))
sun_light.CreateIntensityAttr(3200.0)
sun_light.CreateAngleAttr(0.6)
sun_light.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.9))
UsdGeom.Xformable(sun_light.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))

set_camera_view(
    eye=[1.5, -1.4, 1.35],
    target=[0.0, 0.08, 0.82],
    camera_prim_path="/OmniverseKit_Persp",
)

world.reset()
joystick = init_controller(args.controller)
prev_buttons = []
prev_hats = []

print("[PS5SceneDebug] controls:")
print("[PS5SceneDebug] left stick -> X/Y")
print("[PS5SceneDebug] triggers -> Z up/down")
print("[PS5SceneDebug] button 0 -> reset marker")

try:
    while simulation_app.is_running():
        pygame.event.pump()

        lx = float(joystick.get_axis(0)) if joystick.get_numaxes() > 0 else 0.0
        ly = float(joystick.get_axis(1)) if joystick.get_numaxes() > 1 else 0.0
        l2 = float(joystick.get_axis(4)) if joystick.get_numaxes() > 4 else -1.0
        r2 = float(joystick.get_axis(5)) if joystick.get_numaxes() > 5 else -1.0

        if abs(lx) < args.deadband:
            lx = 0.0
        if abs(ly) < args.deadband:
            ly = 0.0

        l2_norm = max(0.0, min(1.0, (l2 + 1.0) * 0.5))
        r2_norm = max(0.0, min(1.0, (r2 + 1.0) * 0.5))
        z_delta = (r2_norm - l2_norm) * args.translation_scale

        marker_position[0] += lx * args.translation_scale
        marker_position[1] += -ly * args.translation_scale
        marker_position[2] += z_delta

        marker_position[0] = np.clip(marker_position[0], -0.60, 0.60)
        marker_position[1] = np.clip(marker_position[1], -0.25, 0.65)
        marker_position[2] = np.clip(marker_position[2], table_surface_z + 0.03, 1.35)
        set_translate(marker_prim, marker_position)

        buttons = [i for i in range(joystick.get_numbuttons()) if joystick.get_button(i)]
        hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]

        if 0 in buttons and 0 not in prev_buttons:
            marker_position = np.array([0.0, 0.10, table_surface_z + 0.08], dtype=np.float32)
            set_translate(marker_prim, marker_position)
            print("[PS5SceneDebug] reset marker")

        if buttons != prev_buttons or hats != prev_hats:
            print(
                f"[PS5SceneDebug] buttons={buttons or ['NONE']} hats={hats} "
                f"marker=({marker_position[0]:.3f}, {marker_position[1]:.3f}, {marker_position[2]:.3f})"
            )
            prev_buttons = buttons
            prev_hats = hats

        world.step(render=True)
finally:
    joystick.quit()
    pygame.joystick.quit()
    pygame.quit()
    simulation_app.close()
