import os
from dataclasses import dataclass, field
from typing import Any

import omni.usd
import omni.kit.commands
from isaacsim.core.prims import GeometryPrim, SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, UsdPhysics, Vt


@dataclass
class SceneInfo:
    scene_name: str
    robot: SingleArticulation
    robot_prim_path: str
    robot_camera_prim_path: str
    topdown_camera_prim_path: str
    table_surface_z: float
    task_center: tuple[float, float, float]
    task_coverage_xy: tuple[float, float]
    scene_config: dict[str, Any]
    extras: dict[str, Any] = field(default_factory=dict)


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
    mesh.CreatePointsAttr([Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, -0.5, 0.0), Gf.Vec3f(0.5, 0.5, 0.0), Gf.Vec3f(-0.5, 0.5, 0.0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateNormalsAttr([Gf.Vec3f(0.0, 0.0, 1.0)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr([Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, 0.5, 0.0)])
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
    st.Set(Vt.Vec2fArray([Gf.Vec2f(0.0, 0.0), Gf.Vec2f(uv_scale[0], 0.0), Gf.Vec2f(uv_scale[0], uv_scale[1]), Gf.Vec2f(0.0, uv_scale[1])]))
    return mesh.GetPrim()


def create_omnipbr_material(stage, material_path, color=None, roughness=0.5, metallic=0.0, texture_path=None):
    created = []
    omni.kit.commands.execute("CreateAndBindMdlMaterialFromLibrary", mdl_name="OmniPBR.mdl", mtl_name="OmniPBR", mtl_created_list=created)
    material_prim = stage.GetPrimAtPath(created[0])
    if created[0] != material_path:
        omni.kit.commands.execute("MovePrim", path_from=created[0], path_to=material_path)
        material_prim = stage.GetPrimAtPath(material_path)
    if color is not None:
        omni.usd.create_material_input(material_prim, "diffuse_color_constant", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f)
    omni.usd.create_material_input(material_prim, "reflection_roughness_constant", float(roughness), Sdf.ValueTypeNames.Float)
    omni.usd.create_material_input(material_prim, "metallic_constant", float(metallic), Sdf.ValueTypeNames.Float)
    if texture_path:
        omni.usd.create_material_input(material_prim, "diffuse_texture", texture_path, Sdf.ValueTypeNames.Asset)
    return UsdShade.Material(material_prim)


def bind_material(prim, material):
    UsdShade.MaterialBindingAPI(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)


def apply_static_collision(prim_paths):
    for prim_path in prim_paths:
        GeometryPrim(prim_path).apply_collision_apis()


def deactivate_embedded_environment(stage, root_prim_path):
    candidate_tokens = {"environment", "groundplane", "ground_plane", "ground", "floor", "physicsscene", "camerasettings", "viewport_l", "viewport_r", "data"}
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        if prim_path.startswith(root_prim_path + "/") and prim.GetName().lower() in candidate_tokens:
            prim.SetActive(False)


def set_translate(prim, translate):
    tx, ty, tz = float(translate[0]), float(translate[1]), float(translate[2])
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(tx, ty, tz))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))


def create_robot_camera(stage, prim_path):
    mount = UsdGeom.Xform.Define(stage, prim_path)
    mount_xform = UsdGeom.Xformable(mount.GetPrim())
    mount_xform.AddTranslateOp().Set(Gf.Vec3d(0.03, 0.0, 0.0))
    mount_xform.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 90.0))
    camera = UsdGeom.Camera.Define(stage, f"{prim_path}/Sensor")
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 180.0))
    camera.CreateFocalLengthAttr(2.79)
    camera.CreateHorizontalApertureAttr(3.84)
    camera.CreateVerticalApertureAttr(2.88)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 8.0))
    return camera.GetPrim()


def create_topdown_camera(stage, prim_path, center, coverage_xy):
    camera = UsdGeom.Camera.Define(stage, prim_path)
    xform = UsdGeom.Xformable(camera.GetPrim())
    focal_length = 18.0
    horizontal_aperture = 20.955
    vertical_aperture = 15.2908
    required_height_x = float(coverage_xy[0]) * focal_length / horizontal_aperture
    required_height_y = float(coverage_xy[1]) * focal_length / vertical_aperture
    camera_height = 2.0 * (max(required_height_x, required_height_y) + 0.10)
    xform.AddTranslateOp().Set(Gf.Vec3d(float(center[0]), float(center[1]), float(center[2] + camera_height)))
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(horizontal_aperture)
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 12.0))
    return camera.GetPrim()


def build_tray(stage, root_path, center_xy, surface_z, outer_size, height, wall, material):
    outer_w, outer_d = outer_size
    tray_floor_z = surface_z + wall * 0.5
    tray_center_z = surface_z + height * 0.5
    inner_w = outer_w - 2.0 * wall
    inner_d = outer_d - 2.0 * wall
    parts = {
        "Base": ((outer_w, outer_d, wall), (center_xy[0], center_xy[1], tray_floor_z)),
        "WallL": ((wall, outer_d, height), (center_xy[0] - outer_w * 0.5 + wall * 0.5, center_xy[1], tray_center_z)),
        "WallR": ((wall, outer_d, height), (center_xy[0] + outer_w * 0.5 - wall * 0.5, center_xy[1], tray_center_z)),
        "WallB": ((inner_w, wall, height), (center_xy[0], center_xy[1] - outer_d * 0.5 + wall * 0.5, tray_center_z)),
        "WallF": ((inner_w, wall, height), (center_xy[0], center_xy[1] + outer_d * 0.5 - wall * 0.5, tray_center_z)),
    }
    paths = []
    for name, (size, translate) in parts.items():
        prim = define_box(stage, f"{root_path}/{name}", size=size, translate=translate)
        bind_material(prim, material)
        paths.append(str(prim.GetPath()))
    return paths


def build_tabletop_with_cutouts(stage, root_path, table_size, table_top_z, hole_specs, base_material, surface_material):
    table_half_x = table_size[0] * 0.5
    table_half_y = table_size[1] * 0.5
    table_surface_z = table_top_z + table_size[2] * 0.5
    x_breaks = [-table_half_x]
    for center_x, _, hole_w, _ in hole_specs:
        x_breaks.extend([center_x - hole_w * 0.5, center_x + hole_w * 0.5])
    x_breaks.append(table_half_x)
    x_breaks = sorted(set(round(v, 6) for v in x_breaks))
    box_paths = []
    for slab_idx in range(len(x_breaks) - 1):
        x0 = x_breaks[slab_idx]
        x1 = x_breaks[slab_idx + 1]
        slab_width = x1 - x0
        if slab_width <= 1e-6:
            continue
        active_holes = []
        for center_x, center_y, hole_w, hole_d in hole_specs:
            hole_min_x = center_x - hole_w * 0.5
            hole_max_x = center_x + hole_w * 0.5
            if x0 >= hole_min_x - 1e-6 and x1 <= hole_max_x + 1e-6:
                active_holes.append((center_y, hole_d))
        if not active_holes:
            piece = define_box(stage, f"{root_path}/Top_{slab_idx}", size=(slab_width, table_size[1], table_size[2]), translate=((x0 + x1) * 0.5, 0.0, table_top_z))
            bind_material(piece, base_material)
            top_plane = define_uv_plane(stage, f"{root_path}/TopSurface_{slab_idx}", size=(max(0.0, slab_width - 0.01), max(0.0, table_size[1] - 0.01)), translate=((x0 + x1) * 0.5, 0.0, table_surface_z + 0.001))
            bind_material(top_plane, surface_material)
            box_paths.append(str(piece.GetPath()))
            continue
        center_y, hole_d = active_holes[0]
        hole_min_y = center_y - hole_d * 0.5
        hole_max_y = center_y + hole_d * 0.5
        y_segments = [((-table_half_y + hole_min_y) * 0.5, hole_min_y - (-table_half_y)), ((hole_max_y + table_half_y) * 0.5, table_half_y - hole_max_y)]
        for seg_idx, (seg_center_y, seg_depth) in enumerate(y_segments):
            if seg_depth <= 1e-6:
                continue
            piece = define_box(stage, f"{root_path}/Top_{slab_idx}_{seg_idx}", size=(slab_width, seg_depth, table_size[2]), translate=((x0 + x1) * 0.5, seg_center_y, table_top_z))
            bind_material(piece, base_material)
            top_plane = define_uv_plane(stage, f"{root_path}/TopSurface_{slab_idx}_{seg_idx}", size=(max(0.0, slab_width - 0.01), max(0.0, seg_depth - 0.01)), translate=((x0 + x1) * 0.5, seg_center_y, table_surface_z + 0.001))
            bind_material(top_plane, surface_material)
            box_paths.append(str(piece.GetPath()))
    return box_paths


def list_ycb_assets(ycb_dir):
    entries = [os.path.join(ycb_dir, name) for name in sorted(os.listdir(ycb_dir)) if name.endswith(".usd") and not name.startswith(".")]
    if not entries:
        raise RuntimeError(f"No YCB USD files found in: {ycb_dir}")
    return entries


def compute_bbox_range(usd_path):
    temp_stage = Usd.Stage.Open(usd_path)
    if temp_stage is None:
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy], useExtentsHint=True)
    world = Gf.Range3d()
    found = False
    for prim in temp_stage.Traverse():
        if prim.IsA(UsdGeom.Imageable):
            bbox = bbox_cache.ComputeWorldBound(prim)
            rng = bbox.ComputeAlignedRange()
            if not rng.IsEmpty():
                if not found:
                    world = Gf.Range3d(rng.GetMin(), rng.GetMax())
                    found = True
                else:
                    world.UnionWith(rng)
    if not found:
        return None
    return world


def compute_bbox_volume(usd_path):
    rng = compute_bbox_range(usd_path)
    if rng is None:
        return None
    size = rng.GetSize()
    return float(size[0]) * float(size[1]) * float(size[2])


def find_meshes(root_prim):
    return [prim for prim in Usd.PrimRange(root_prim) if prim.IsA(UsdGeom.Mesh)]


def auto_generate_convex_colliders(root_prim):
    meshes = find_meshes(root_prim)
    for mesh in meshes:
        UsdPhysics.CollisionAPI.Apply(mesh)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(mesh)
        mesh_api.CreateApproximationAttr().Set("convexHull")
    return meshes


def offset_prim_translate_z(stage, prim_path, delta_z):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    target_prim = prim.GetParent()
    if not target_prim.IsValid():
        target_prim = prim
    xformable = UsdGeom.Xformable(target_prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if not translate_ops:
        return
    current = translate_ops[0].Get()
    translate_ops[0].Set(Gf.Vec3d(float(current[0]), float(current[1]), float(current[2] + delta_z)))


def add_robot(stage, robot_usd, robot_x, robot_y, robot_yaw, table_surface_z):
    mount_prim = UsdGeom.Xform.Define(stage, "/World/RobotMount/PiperX")
    mount_xform = UsdGeom.Xformable(mount_prim.GetPrim())
    mount_xform.AddTranslateOp().Set(Gf.Vec3d(robot_x, robot_y, table_surface_z))
    mount_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, robot_yaw))
    add_reference_to_stage(usd_path=os.path.abspath(robot_usd), prim_path="/World/RobotMount/PiperX/Robot")
    deactivate_embedded_environment(stage, "/World/RobotMount/PiperX/Robot")
    return SingleArticulation(prim_path="/World/RobotMount/PiperX/Robot", name="piper_x")
