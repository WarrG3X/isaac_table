from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import os

import numpy as np
import omni.kit.commands
from isaacsim.core.api import World
from isaacsim.core.prims import GeometryPrim, SingleRigidPrim
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, UsdPhysics, PhysxSchema


ASSET_ROOT = r"C:\Users\Warra\Downloads\Assets\Isaac\5.1"
AXIS_ALIGNED = os.path.join(ASSET_ROOT, "Isaac", "Props", "YCB", "Axis_Aligned", "003_cracker_box.usd")
AXIS_ALIGNED_PHYSICS = os.path.join(ASSET_ROOT, "Isaac", "Props", "YCB", "Axis_Aligned_Physics", "003_cracker_box.usd")


def define_box(stage, prim_path, size, translate):
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp().Set(Gf.Vec3d(*translate))
    cube = UsdGeom.Cube.Define(stage, f"{prim_path}/Geom")
    cube.CreateSizeAttr(1.0)
    UsdGeom.Xformable(cube.GetPrim()).AddScaleOp().Set(Gf.Vec3f(*size))
    return cube.GetPrim()


def create_omnipbr_material(stage, material_path, color=None, roughness=0.5):
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
        omni.usd.create_material_input(material_prim, "diffuse_color_constant", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f)
    omni.usd.create_material_input(material_prim, "reflection_roughness_constant", float(roughness), Sdf.ValueTypeNames.Float)
    return UsdShade.Material(material_prim)


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def apply_static_collision(prim_path):
    GeometryPrim(prim_path).apply_collision_apis()


def describe_physics(stage, prim_path, label):
    root = stage.GetPrimAtPath(prim_path)
    print(f"[YCBProbe] {label}: {prim_path}")
    if not root.IsValid():
        print("  invalid prim")
        return
    physics_hits = []
    for prim in Usd.PrimRange(root):
        props = list(prim.GetPropertyNames())
        if any(
            (
                "physics" in p.lower()
                or "physx" in p.lower()
                or "collision" in p.lower()
                or "rigid" in p.lower()
            )
            for p in props
        ):
            physics_hits.append((str(prim.GetPath()), [p for p in props if "physics" in p.lower() or "physx" in p.lower() or "collision" in p.lower() or "rigid" in p.lower()][:10]))
    print(f"  physicsish prims: {len(physics_hits)}")
    for path, props in physics_hits[:10]:
        print(f"   {path} :: {props}")


from pxr import Usd

world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Floor")
UsdGeom.Xform.Define(stage, "/World/Test")
UsdGeom.Xform.Define(stage, "/World/Lights")

floor_material = create_omnipbr_material(stage, "/World/Looks/Floor", color=(0.82, 0.82, 0.84), roughness=0.65)
platform_material = create_omnipbr_material(stage, "/World/Looks/Platform", color=(0.40, 0.42, 0.46), roughness=0.58)

floor = define_box(stage, "/World/Floor/Base", size=(4.0, 4.0, 0.02), translate=(0.0, 0.0, -0.01))
bind_material(floor, floor_material)
platform = define_box(stage, "/World/Test/Platform", size=(1.0, 0.5, 0.04), translate=(0.0, 0.0, 0.20))
bind_material(platform, platform_material)
apply_static_collision(str(floor.GetPath()))
apply_static_collision(str(platform.GetPath()))

dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
dome.CreateIntensityAttr(850.0)
sun = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Sun"))
sun.CreateIntensityAttr(2500.0)
UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 35.0))

add_reference_to_stage(usd_path=AXIS_ALIGNED, prim_path="/World/Test/AxisAligned")
add_reference_to_stage(usd_path=AXIS_ALIGNED_PHYSICS, prim_path="/World/Test/AxisAlignedPhysics")

axis_aligned = world.scene.add(
    SingleRigidPrim(
        prim_path="/World/Test/AxisAligned",
        name="axis_aligned",
        position=np.array([-0.18, 0.0, 0.55], dtype=np.float32),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
)
axis_aligned_physics = world.scene.add(
    SingleRigidPrim(
        prim_path="/World/Test/AxisAlignedPhysics",
        name="axis_aligned_physics",
        position=np.array([0.18, 0.0, 0.55], dtype=np.float32),
        orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
)

set_camera_view(
    eye=[1.5, 1.3, 1.2],
    target=[0.0, 0.0, 0.35],
    camera_prim_path="/OmniverseKit_Persp",
)

world.reset()
axis_aligned.initialize()
axis_aligned_physics.initialize()

describe_physics(stage, "/World/Test/AxisAligned", "Axis_Aligned")
describe_physics(stage, "/World/Test/AxisAlignedPhysics", "Axis_Aligned_Physics")

for _ in range(120):
    world.step(render=True)

aa_pos = np.asarray(axis_aligned.get_world_pose()[0], dtype=np.float32)
aap_pos = np.asarray(axis_aligned_physics.get_world_pose()[0], dtype=np.float32)
print(f"[YCBProbe] Axis_Aligned final position: {aa_pos}")
print(f"[YCBProbe] Axis_Aligned_Physics final position: {aap_pos}")

while simulation_app.is_running():
    world.step(render=True)

simulation_app.close()
