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
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade, Vt
from isaacsim.core.api import World
from isaacsim.core.prims import GeometryPrim
from isaacsim.core.utils.viewports import set_camera_view


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

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


WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT,
    wintypes.HWND,
    wintypes.UINT,
    wintypes.WPARAM,
    wintypes.LPARAM,
)


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
        self.class_name = "IsaacSpaceMouseRawInputWindow"
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

        hwnd = user32.CreateWindowExW(
            0,
            self.class_name,
            self.class_name,
            0,
            0,
            0,
            0,
            0,
            None,
            None,
            self.hinstance,
            None,
        )
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
        ok = user32.RegisterRawInputDevices(
            ctypes.byref(raw_input_device),
            1,
            ctypes.sizeof(RAWINPUTDEVICE),
        )
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


def set_translate(prim, translate):
    tx, ty, tz = float(translate[0]), float(translate[1]), float(translate[2])
    xformable = UsdGeom.Xformable(prim)
    translate_ops = [op for op in xformable.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    if translate_ops:
        translate_ops[0].Set(Gf.Vec3d(tx, ty, tz))
    else:
        xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, tz))


parser = argparse.ArgumentParser()
parser.add_argument("--vendor-id", default="256f", help="SpaceMouse USB vendor ID in hex.")
parser.add_argument("--product-id", default="c635", help="SpaceMouse USB product ID in hex.")
parser.add_argument("--translation-scale", type=float, default=0.00018, help="Meters per raw input unit per frame.")
parser.add_argument("--deadband", type=float, default=40.0, help="Deadband applied to raw translational axes.")
args, _ = parser.parse_known_args()

world = World(stage_units_in_meters=1.0)
stage = world.stage
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.Xform.Define(stage, "/World")
UsdGeom.Xform.Define(stage, "/World/Looks")
UsdGeom.Xform.Define(stage, "/World/Floor")
UsdGeom.Xform.Define(stage, "/World/Table")
UsdGeom.Xform.Define(stage, "/World/Marker")
UsdGeom.Xform.Define(stage, "/World/Lights")

bamboo_texture_path = os.path.join(SCRIPT_DIR, "materials", "nv_bamboo_desktop.jpg")
granite_texture_path = os.path.join(SCRIPT_DIR, "materials", "nv_granite_tile.jpg")

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
marker_material = create_omnipbr_material(stage, "/World/Looks/Marker", color=(0.84, 0.18, 0.12), roughness=0.2)

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

leg_height = 0.72
table_leg_paths = []
for name, position in {
    "LegFL": (0.62, 0.32, leg_height * 0.5),
    "LegFR": (0.62, -0.32, leg_height * 0.5),
    "LegBL": (-0.62, 0.32, leg_height * 0.5),
    "LegBR": (-0.62, -0.32, leg_height * 0.5),
}.items():
    leg_prim = define_box(stage, f"/World/Table/{name}", size=(0.06, 0.06, leg_height), translate=position)
    bind_material(leg_prim, leg_material)
    table_leg_paths.append(str(leg_prim.GetPath()))

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

marker = UsdGeom.Sphere.Define(stage, "/World/Marker/Probe")
marker.CreateRadiusAttr(0.04)
marker_prim = marker.GetPrim()
bind_material(marker_prim, marker_material)
set_translate(marker_prim, np.array([0.0, 0.0, table_surface_z + 0.12], dtype=np.float32))

set_camera_view(
    eye=[2.3, -2.0, 1.6],
    target=[0.0, 0.0, 0.82],
    camera_prim_path="/OmniverseKit_Persp",
)

world.reset()

reader = SpaceMouseReader(args.vendor_id, args.product_id)
reader.start_in_thread()

marker_position = np.array([0.0, 0.0, table_surface_z + 0.12], dtype=np.float32)
last_button_state = 0


def handle_signal(_signum, _frame):
    reader.stop()


signal.signal(signal.SIGINT, handle_signal)
print("[SpaceMouse] debug scene ready")
print("[SpaceMouse] translation controls marker X/Y/Z over the table")

while simulation_app.is_running():
    snapshot = reader.snapshot()
    tx, ty, tz = snapshot["translation"]
    buttons = snapshot["buttons"]

    trans = np.array([tx, ty, tz], dtype=np.float32)
    trans[np.abs(trans) < args.deadband] = 0.0

    # Map SpaceMouse translation to scene world axes.
    delta = np.array(
        [
            trans[0] * args.translation_scale,
            -trans[1] * args.translation_scale,
            -trans[2] * args.translation_scale,
        ],
        dtype=np.float32,
    )
    marker_position += delta
    marker_position[0] = np.clip(marker_position[0], -0.55, 0.55)
    marker_position[1] = np.clip(marker_position[1], -0.30, 0.30)
    marker_position[2] = np.clip(marker_position[2], table_surface_z + 0.03, 1.45)
    set_translate(marker_prim, marker_position)

    if buttons != last_button_state:
        print(f"[SpaceMouse] buttons=0x{buttons:02x} marker=({marker_position[0]:.3f}, {marker_position[1]:.3f}, {marker_position[2]:.3f})")
        last_button_state = buttons

    world.step(render=True)

reader.stop()
simulation_app.close()
