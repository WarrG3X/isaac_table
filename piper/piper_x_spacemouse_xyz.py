from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import argparse
import ctypes
import os
import signal
import struct
import threading
import time
from ctypes import wintypes

import numpy as np
import omni.kit.commands
import omni.ui as ui
from isaacsim.core.api import World
from isaacsim.core.prims import GeometryPrim, SingleArticulation
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats, rot_matrices_to_quats
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot_motion.motion_generation import ArticulationKinematicsSolver, LulaKinematicsSolver
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
ISAAC_ROOT = os.path.abspath(os.path.join(REPO_ROOT, "..", "..", ".."))
DEFAULT_PIPER_USD = os.path.join(ISAAC_ROOT, "piper_isaac_sim", "USD", "piper_x_v1.usd")
DEFAULT_LULA_YAML = os.path.join(REPO_ROOT, "desc", "piper_x_robot_description.yaml")
DEFAULT_LULA_URDF = os.path.join(
    ISAAC_ROOT, "piper_isaac_sim", "piper_x_description", "urdf", "piper_x_description_d435.urdf"
)

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

LRESULT = ctypes.c_ssize_t
HRAWINPUT = wintypes.HANDLE
HINSTANCE = wintypes.HANDLE
HICON = wintypes.HANDLE
HCURSOR = wintypes.HANDLE
HBRUSH = wintypes.HANDLE
HWND = wintypes.HWND
UINT = wintypes.UINT
WPARAM = wintypes.WPARAM
LPARAM = wintypes.LPARAM

user32.DefWindowProcW.argtypes = [HWND, UINT, WPARAM, LPARAM]
user32.DefWindowProcW.restype = LRESULT

WM_DESTROY = 0x0002
WM_INPUT = 0x00FF

RIDEV_INPUTSINK = 0x00000100
RID_INPUT = 0x10000003
RIDI_DEVICENAME = 0x20000007
RIM_TYPEHID = 2

TARGET_USAGE_PAGE = 0x01
TARGET_USAGE = 0x08


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wintypes.USHORT),
        ("usUsage", wintypes.USHORT),
        ("dwFlags", wintypes.DWORD),
        ("hwndTarget", wintypes.HWND),
    ]


class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wintypes.DWORD),
        ("dwSize", wintypes.DWORD),
        ("hDevice", wintypes.HANDLE),
        ("wParam", wintypes.WPARAM),
    ]


class RAWHID(ctypes.Structure):
    _fields_ = [
        ("dwSizeHid", wintypes.DWORD),
        ("dwCount", wintypes.DWORD),
    ]


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", wintypes.LONG),
        ("y", wintypes.LONG),
    ]


class MSG(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", POINT),
        ("lPrivate", wintypes.DWORD),
    ]


WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)


class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", HINSTANCE),
        ("hIcon", HICON),
        ("hCursor", HCURSOR),
        ("hbrBackground", HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
    ]


class SpaceMouseReader:
    def __init__(self, vendor_id: str, product_id: str):
        self.vendor_id = vendor_id.lower()
        self.product_id = product_id.lower()
        self.class_name = "IsaacPiperSpaceMouseWindow"
        self.device_names = {}
        self.running = False
        self.lock = threading.Lock()
        self.state = {
            "translation": (0, 0, 0),
            "rotation": (0, 0, 0),
            "buttons": 0,
            "updated_at": time.time(),
        }
        self.wndproc = WNDPROC(self._wndproc)
        self.hinstance = kernel32.GetModuleHandleW(None)
        self.hwnd = None
        self.thread = None

    def start_in_thread(self):
        if self.thread is not None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.hwnd:
            user32.PostMessageW(self.hwnd, WM_DESTROY, 0, 0)
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def snapshot(self):
        with self.lock:
            return {
                "translation": tuple(self.state["translation"]),
                "rotation": tuple(self.state["rotation"]),
                "buttons": int(self.state["buttons"]),
                "updated_at": float(self.state["updated_at"]),
            }

    def _run(self):
        self._create_window()
        self._register_raw_input()
        print(f"[SpaceMouse] listening for {self.vendor_id}:{self.product_id}")
        self._message_loop()

    def _create_window(self):
        wndclass = WNDCLASS()
        wndclass.lpfnWndProc = self.wndproc
        wndclass.hInstance = self.hinstance
        wndclass.lpszClassName = self.class_name
        atom = user32.RegisterClassW(ctypes.byref(wndclass))
        if not atom and ctypes.get_last_error() != 1410:
            raise ctypes.WinError(ctypes.get_last_error())
        hwnd = user32.CreateWindowExW(0, self.class_name, self.class_name, 0, 0, 0, 0, 0, None, None, self.hinstance, None)
        if not hwnd:
            raise ctypes.WinError(ctypes.get_last_error())
        self.hwnd = hwnd

    def _register_raw_input(self):
        raw_input_device = RAWINPUTDEVICE(
            usUsagePage=TARGET_USAGE_PAGE,
            usUsage=TARGET_USAGE,
            dwFlags=RIDEV_INPUTSINK,
            hwndTarget=self.hwnd,
        )
        ok = user32.RegisterRawInputDevices(ctypes.byref(raw_input_device), 1, ctypes.sizeof(RAWINPUTDEVICE))
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    def _message_loop(self):
        msg = MSG()
        while self.running:
            result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result == -1:
                raise ctypes.WinError(ctypes.get_last_error())
            if result == 0:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _wndproc(self, hwnd, msg, wparam, lparam):
        if msg == WM_INPUT:
            self._handle_raw_input(lparam)
            return 0
        if msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(HWND(hwnd), UINT(msg), WPARAM(wparam), LPARAM(lparam))

    def _handle_raw_input(self, lparam):
        size = wintypes.UINT(0)
        header_size = ctypes.sizeof(RAWINPUTHEADER)
        result = user32.GetRawInputData(HRAWINPUT(lparam), RID_INPUT, None, ctypes.byref(size), header_size)
        if result == 0xFFFFFFFF or not size.value:
            return
        raw_buffer = ctypes.create_string_buffer(size.value)
        result = user32.GetRawInputData(HRAWINPUT(lparam), RID_INPUT, raw_buffer, ctypes.byref(size), header_size)
        if result == 0xFFFFFFFF:
            return
        header = RAWINPUTHEADER.from_buffer_copy(raw_buffer[:header_size])
        if header.dwType != RIM_TYPEHID:
            return
        if not self._device_matches(header.hDevice):
            return
        hid_offset = header_size
        hid_header_size = ctypes.sizeof(RAWHID)
        hid = RAWHID.from_buffer_copy(raw_buffer[hid_offset : hid_offset + hid_header_size])
        data_offset = hid_offset + hid_header_size
        total = hid.dwSizeHid * hid.dwCount
        payload = raw_buffer[data_offset : data_offset + total]
        for index in range(hid.dwCount):
            start = index * hid.dwSizeHid
            end = start + hid.dwSizeHid
            self._decode_report(bytes(payload[start:end]))

    def _device_matches(self, device_handle):
        cached = self.device_names.get(device_handle)
        if cached is None:
            size = wintypes.UINT(0)
            user32.GetRawInputDeviceInfoW(device_handle, RIDI_DEVICENAME, None, ctypes.byref(size))
            if not size.value:
                self.device_names[device_handle] = ""
                return False
            name_buffer = ctypes.create_unicode_buffer(size.value)
            result = user32.GetRawInputDeviceInfoW(device_handle, RIDI_DEVICENAME, name_buffer, ctypes.byref(size))
            if result == 0xFFFFFFFF:
                self.device_names[device_handle] = ""
                return False
            cached = name_buffer.value.lower()
            self.device_names[device_handle] = cached
        return f"vid_{self.vendor_id}" in cached and f"pid_{self.product_id}" in cached

    def _decode_report(self, report: bytes):
        if not report:
            return
        with self.lock:
            report_id = report[0]
            if report_id == 1 and len(report) >= 7:
                self.state["translation"] = struct.unpack("<3h", report[1:7])
                self.state["updated_at"] = time.time()
                return
            if report_id == 2 and len(report) >= 7:
                self.state["rotation"] = struct.unpack("<3h", report[1:7])
                self.state["updated_at"] = time.time()
                return
            if report_id == 3 and len(report) >= 2:
                self.state["buttons"] = report[1]
                self.state["updated_at"] = time.time()


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
        [Gf.Vec3f(-0.5, -0.5, 0.0), Gf.Vec3f(0.5, -0.5, 0.0), Gf.Vec3f(0.5, 0.5, 0.0), Gf.Vec3f(-0.5, 0.5, 0.0)]
    )
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
    mtl_created_list = []
    omni.kit.commands.execute("CreateAndBindMdlMaterialFromLibrary", mdl_name="OmniPBR.mdl", mtl_name="OmniPBR", mtl_created_list=mtl_created_list)
    material_prim = stage.GetPrimAtPath(mtl_created_list[0])
    if mtl_created_list[0] != material_path:
        omni.kit.commands.execute("MovePrim", path_from=mtl_created_list[0], path_to=material_path)
        material_prim = stage.GetPrimAtPath(material_path)
    if color is not None:
        omni.usd.create_material_input(material_prim, "diffuse_color_constant", Gf.Vec3f(*color), Sdf.ValueTypeNames.Color3f)
    omni.usd.create_material_input(material_prim, "reflection_roughness_constant", roughness, Sdf.ValueTypeNames.Float)
    omni.usd.create_material_input(material_prim, "metallic_constant", metallic, Sdf.ValueTypeNames.Float)
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


def set_translate(prim, translate):
    tx, ty, tz = float(translate[0]), float(translate[1]), float(translate[2])
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(tx, ty, tz))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))


def build_status_window(initial_position):
    text_model = ui.SimpleStringModel("")
    window = ui.Window("Piper SpaceMouse XYZ", width=430, height=120, visible=True)
    window.position_x = 40
    window.position_y = 60
    with window.frame:
        with ui.VStack(spacing=8, height=0):
            ui.Label("Translation-only EE teleop. Orientation stays fixed.")
            with ui.HStack(height=24):
                ui.Label("Target", width=50)
                ui.StringField(model=text_model, read_only=True)
    text_model.set_value(f"({initial_position[0]:.3f}, {initial_position[1]:.3f}, {initial_position[2]:.3f})")
    return window, text_model


def quat_multiply_wxyz(q1, q2):
    w1, x1, y1, z1 = [float(v) for v in q1]
    w2, x2, y2, z2 = [float(v) for v in q2]
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / norm


parser = argparse.ArgumentParser()
parser.add_argument("--vendor-id", default="256f")
parser.add_argument("--product-id", default="c635")
parser.add_argument("--translation-scale", type=float, default=0.00006)
parser.add_argument("--rotation-scale", type=float, default=0.00008)
parser.add_argument("--deadband", type=float, default=40.0)
parser.add_argument("--robot-usd", type=str, default=DEFAULT_PIPER_USD)
parser.add_argument("--lula-yaml", type=str, default=DEFAULT_LULA_YAML)
parser.add_argument("--lula-urdf", type=str, default=DEFAULT_LULA_URDF)
parser.add_argument("--robot-x", type=float, default=0.0)
parser.add_argument("--robot-y", type=float, default=0.12)
parser.add_argument("--robot-yaw", type=float, default=90.0)
parser.add_argument("--ee-frame", type=str, default="Link6")
args, _ = parser.parse_known_args()

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
UsdGeom.Xform.Define(stage, "/World/Debug")

bamboo_texture_path = os.path.join(REPO_ROOT, "materials", "nv_bamboo_desktop.jpg")
granite_texture_path = os.path.join(REPO_ROOT, "materials", "nv_granite_tile.jpg")
aluminum_anodized_dir = os.path.join(REPO_ROOT, "materials", "Aluminum_Anodized")
aluminum_anodized_basecolor = os.path.join(aluminum_anodized_dir, "Aluminum_Anodized_BaseColor.png")
aluminum_anodized_normal = os.path.join(aluminum_anodized_dir, "Aluminum_Anodized_Normal.png")
aluminum_anodized_orm = os.path.join(aluminum_anodized_dir, "Aluminum_Anodized_ORM.png")

floor_material_base = create_omnipbr_material(stage, "/World/Looks/FloorBase", color=(0.44, 0.46, 0.48), roughness=0.72)
floor_material = create_omnipbr_material(stage, "/World/Looks/Floor", color=(0.85, 0.85, 0.85), roughness=0.62, texture_path=granite_texture_path if os.path.exists(granite_texture_path) else None)
wood_material = create_omnipbr_material(stage, "/World/Looks/Wood", color=(0.72, 0.66, 0.54), roughness=0.38, texture_path=bamboo_texture_path if os.path.exists(bamboo_texture_path) else None)
wood_base_material = create_omnipbr_material(stage, "/World/Looks/WoodBase", color=(0.49, 0.35, 0.20), roughness=0.48)
leg_material = create_omnipbr_material(stage, "/World/Looks/Legs", color=(0.19, 0.12, 0.08), roughness=0.5)
bin_material = create_omnipbr_material(stage, "/World/Looks/Bin", color=(0.50, 0.50, 0.50), roughness=0.0, metallic=0.0, texture_path=aluminum_anodized_basecolor if os.path.exists(aluminum_anodized_basecolor) else None)
target_material = create_omnipbr_material(stage, "/World/Looks/Target", color=(0.82, 0.16, 0.12), roughness=0.25)

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
robot = SingleArticulation(prim_path="/World/RobotMount/PiperX/Robot", name="piper_x")

target_marker = UsdGeom.Sphere.Define(stage, "/World/Debug/TeleopTarget")
target_marker.CreateRadiusAttr(0.018)
bind_material(target_marker.GetPrim(), target_material)

set_camera_view(eye=[1.7, -1.6, 1.45], target=[0.0, 0.15, 0.88], camera_prim_path="/OmniverseKit_Persp")

world.reset()
robot.initialize()
articulation_controller = robot.get_articulation_controller()
lula_joint_names = list(robot.dof_names)
dof_props = robot.dof_properties
joint7_idx = lula_joint_names.index("joint7")
joint8_idx = lula_joint_names.index("joint8")
gripper_open_target = np.array(
    [float(dof_props["upper"][joint7_idx]), float(dof_props["lower"][joint8_idx])],
    dtype=np.float32,
)
gripper_closed_target = np.array(
    [float(dof_props["lower"][joint7_idx]), float(dof_props["upper"][joint8_idx])],
    dtype=np.float32,
)
lula_solver = LulaKinematicsSolver(robot_description_path=os.path.abspath(args.lula_yaml), urdf_path=os.path.abspath(args.lula_urdf))
robot_base_position = np.array([args.robot_x, args.robot_y, table_surface_z], dtype=np.float64)
robot_base_orientation = euler_angles_to_quats(np.array([0.0, 0.0, args.robot_yaw]), degrees=True)
lula_solver.set_robot_base_pose(robot_base_position, robot_base_orientation)
art_kinematics = ArticulationKinematicsSolver(robot, lula_solver, args.ee_frame)
ee_position, ee_rotation = art_kinematics.compute_end_effector_pose()
target_position = np.array(ee_position, dtype=np.float32)
target_orientation = rot_matrices_to_quats(np.asarray(ee_rotation))
set_translate(target_marker.GetPrim(), target_position)
status_window, status_model = build_status_window(target_position)
print(f"[PiperX] SpaceMouse XYZ teleop using frame: {args.ee_frame}")
print(f"[PiperX] initial target: {target_position}")
print(f"[PiperX] gripper open target: joint7={gripper_open_target[0]:.5f} joint8={gripper_open_target[1]:.5f}")
print(f"[PiperX] gripper closed target: joint7={gripper_closed_target[0]:.5f} joint8={gripper_closed_target[1]:.5f}")

reader = SpaceMouseReader(args.vendor_id, args.product_id)
reader.start_in_thread()
last_buttons = 0
gripper_open = True


def handle_signal(_signum, _frame):
    reader.stop()


signal.signal(signal.SIGINT, handle_signal)

while simulation_app.is_running():
    snapshot = reader.snapshot()
    tx, ty, tz = snapshot["translation"]
    rx, ry, rz = snapshot["rotation"]
    buttons = snapshot["buttons"]
    trans = np.array([tx, ty, tz], dtype=np.float32)
    rot = np.array([rx, ry, rz], dtype=np.float32)
    trans[np.abs(trans) < args.deadband] = 0.0
    rot[np.abs(rot) < args.deadband] = 0.0
    delta = np.array(
        [
            trans[0] * args.translation_scale,
            -trans[1] * args.translation_scale,
            -trans[2] * args.translation_scale,
        ],
        dtype=np.float32,
    )
    target_position += delta
    rot_delta = np.array(
        [
            rot[0] * args.rotation_scale,
            rot[1] * args.rotation_scale,
            rot[2] * args.rotation_scale,
        ],
        dtype=np.float64,
    )
    if np.any(rot_delta != 0.0):
        delta_quat = euler_angles_to_quats(rot_delta, degrees=False)
        target_orientation = normalize_quat_wxyz(quat_multiply_wxyz(delta_quat, target_orientation))
    target_position[0] = np.clip(target_position[0], -0.60, 0.60)
    target_position[1] = np.clip(target_position[1], -0.20, 0.85)
    target_position[2] = np.clip(target_position[2], table_surface_z + 0.03, 1.35)
    set_translate(target_marker.GetPrim(), target_position)
    status_model.set_value(f"({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f})")

    if (buttons & 0x01) and not (last_buttons & 0x01):
        gripper_open = not gripper_open
        print(f"[SpaceMouse] gripper toggled -> {'open' if gripper_open else 'closed'}")

    action, success = art_kinematics.compute_inverse_kinematics(target_position=target_position, target_orientation=target_orientation)
    if success:
        joint_positions = np.array(action.joint_positions, dtype=np.float32).copy()
        if joint_positions.shape[0] >= 8:
            finger_targets = gripper_open_target if gripper_open else gripper_closed_target
            joint_positions[joint7_idx] = finger_targets[0]
            joint_positions[joint8_idx] = finger_targets[1]
            action = ArticulationAction(joint_positions=joint_positions)
        articulation_controller.apply_action(action)

    if buttons != last_buttons:
        print(f"[SpaceMouse] buttons=0x{buttons:02x} target=({target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}) ik={success}")
        last_buttons = buttons

    world.step(render=True)

reader.stop()
simulation_app.close()
