import argparse
import ctypes
import sys
import time
from ctypes import wintypes


ERROR_SUCCESS = 0
ERROR_DEVICE_NOT_CONNECTED = 1167

XINPUT_GAMEPAD_DPAD_UP = 0x0001
XINPUT_GAMEPAD_DPAD_DOWN = 0x0002
XINPUT_GAMEPAD_DPAD_LEFT = 0x0004
XINPUT_GAMEPAD_DPAD_RIGHT = 0x0008
XINPUT_GAMEPAD_START = 0x0010
XINPUT_GAMEPAD_BACK = 0x0020
XINPUT_GAMEPAD_LEFT_THUMB = 0x0040
XINPUT_GAMEPAD_RIGHT_THUMB = 0x0080
XINPUT_GAMEPAD_LEFT_SHOULDER = 0x0100
XINPUT_GAMEPAD_RIGHT_SHOULDER = 0x0200
XINPUT_GAMEPAD_A = 0x1000
XINPUT_GAMEPAD_B = 0x2000
XINPUT_GAMEPAD_X = 0x4000
XINPUT_GAMEPAD_Y = 0x8000


BUTTON_NAMES = {
    XINPUT_GAMEPAD_DPAD_UP: "DPAD_UP",
    XINPUT_GAMEPAD_DPAD_DOWN: "DPAD_DOWN",
    XINPUT_GAMEPAD_DPAD_LEFT: "DPAD_LEFT",
    XINPUT_GAMEPAD_DPAD_RIGHT: "DPAD_RIGHT",
    XINPUT_GAMEPAD_START: "START",
    XINPUT_GAMEPAD_BACK: "BACK",
    XINPUT_GAMEPAD_LEFT_THUMB: "L3",
    XINPUT_GAMEPAD_RIGHT_THUMB: "R3",
    XINPUT_GAMEPAD_LEFT_SHOULDER: "LB",
    XINPUT_GAMEPAD_RIGHT_SHOULDER: "RB",
    XINPUT_GAMEPAD_A: "A",
    XINPUT_GAMEPAD_B: "B",
    XINPUT_GAMEPAD_X: "X",
    XINPUT_GAMEPAD_Y: "Y",
}


class XINPUT_GAMEPAD(ctypes.Structure):
    _fields_ = [
        ("wButtons", ctypes.c_ushort),
        ("bLeftTrigger", ctypes.c_ubyte),
        ("bRightTrigger", ctypes.c_ubyte),
        ("sThumbLX", ctypes.c_short),
        ("sThumbLY", ctypes.c_short),
        ("sThumbRX", ctypes.c_short),
        ("sThumbRY", ctypes.c_short),
    ]


class XINPUT_STATE(ctypes.Structure):
    _fields_ = [
        ("dwPacketNumber", ctypes.c_ulong),
        ("Gamepad", XINPUT_GAMEPAD),
    ]


def normalize_stick(value):
    return max(-1.0, min(1.0, value / 32767.0))


def normalize_trigger(value):
    return value / 255.0


def button_list(mask):
    return [name for bit, name in BUTTON_NAMES.items() if mask & bit]


def load_xinput():
    for dll_name in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
        try:
            dll = ctypes.WinDLL(dll_name)
            dll.XInputGetState.argtypes = [ctypes.c_uint, ctypes.POINTER(XINPUT_STATE)]
            dll.XInputGetState.restype = ctypes.c_uint
            return dll, dll_name
        except Exception:
            continue
    return None, None


def run_xinput(controller_index: int, poll_hz: float, print_all: bool, deadband: float):
    xinput, dll_name = load_xinput()
    if xinput is None:
        print("[PS5Debug] no XInput DLL available")
        return 1

    print(f"[PS5Debug] backend=xinput dll={dll_name}")
    print(f"[PS5Debug] polling controller index {controller_index}")
    previous_packet = None
    previous_snapshot = None
    interval = 1.0 / max(poll_hz, 1.0)

    try:
        while True:
            state = XINPUT_STATE()
            result = xinput.XInputGetState(controller_index, ctypes.byref(state))
            if result == ERROR_DEVICE_NOT_CONNECTED:
                if previous_packet is not None:
                    print("[PS5Debug] controller disconnected")
                    previous_packet = None
                    previous_snapshot = None
                time.sleep(0.5)
                continue
            if result != ERROR_SUCCESS:
                print(f"[PS5Debug] XInputGetState error: {result}")
                time.sleep(0.5)
                continue

            gamepad = state.Gamepad
            snapshot = {
                "buttons": gamepad.wButtons,
                "lt": normalize_trigger(gamepad.bLeftTrigger),
                "rt": normalize_trigger(gamepad.bRightTrigger),
                "lx": normalize_stick(gamepad.sThumbLX),
                "ly": normalize_stick(gamepad.sThumbLY),
                "rx": normalize_stick(gamepad.sThumbRX),
                "ry": normalize_stick(gamepad.sThumbRY),
            }
            for key in ("lx", "ly", "rx", "ry"):
                if abs(snapshot[key]) < deadband:
                    snapshot[key] = 0.0

            changed = state.dwPacketNumber != previous_packet or snapshot != previous_snapshot
            if print_all or changed:
                print(
                    "[PS5Debug] "
                    f"buttons={button_list(snapshot['buttons']) or ['NONE']} "
                    f"lt={snapshot['lt']:.2f} rt={snapshot['rt']:.2f} "
                    f"lx={snapshot['lx']:.2f} ly={snapshot['ly']:.2f} "
                    f"rx={snapshot['rx']:.2f} ry={snapshot['ry']:.2f}"
                )
                previous_packet = state.dwPacketNumber
                previous_snapshot = snapshot
            time.sleep(interval)
    except KeyboardInterrupt:
        print("[PS5Debug] stopping")
        return 0


def run_pygame(controller_index: int, poll_hz: float, print_all: bool, deadband: float):
    import pygame

    pygame.init()
    pygame.joystick.init()
    count = pygame.joystick.get_count()
    print(f"[PS5Debug] backend=pygame joysticks={count}")
    if count == 0:
        print("[PS5Debug] no joystick detected by pygame")
        return 1
    if controller_index >= count:
        print(f"[PS5Debug] controller index {controller_index} out of range")
        return 1

    joystick = pygame.joystick.Joystick(controller_index)
    joystick.init()
    print(f"[PS5Debug] name={joystick.get_name()}")
    print(f"[PS5Debug] axes={joystick.get_numaxes()} buttons={joystick.get_numbuttons()} hats={joystick.get_numhats()}")

    interval = 1.0 / max(poll_hz, 1.0)
    previous = None

    try:
        while True:
            pygame.event.pump()
            axes = []
            for i in range(joystick.get_numaxes()):
                value = joystick.get_axis(i)
                if abs(value) < deadband:
                    value = 0.0
                axes.append(round(float(value), 4))
            buttons = [i for i in range(joystick.get_numbuttons()) if joystick.get_button(i)]
            hats = [joystick.get_hat(i) for i in range(joystick.get_numhats())]
            snapshot = (tuple(axes), tuple(buttons), tuple(hats))
            if print_all or snapshot != previous:
                print(f"[PS5Debug] axes={axes} buttons={buttons or ['NONE']} hats={hats}")
                previous = snapshot
            time.sleep(interval)
    except KeyboardInterrupt:
        print("[PS5Debug] stopping")
        return 0
    finally:
        joystick.quit()
        pygame.joystick.quit()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=int, default=0, help="Controller index.")
    parser.add_argument("--poll-hz", type=float, default=60.0, help="Polling rate.")
    parser.add_argument("--print-all", action="store_true", help="Print every poll instead of only changes.")
    parser.add_argument("--deadband", type=float, default=0.08, help="Stick deadband after normalization.")
    parser.add_argument("--backend", choices=["auto", "pygame", "xinput"], default="auto")
    args = parser.parse_args()

    if args.backend in ("auto", "pygame"):
        try:
            import pygame  # noqa: F401
            return run_pygame(args.controller, args.poll_hz, args.print_all, args.deadband)
        except Exception as e:
            if args.backend == "pygame":
                print(f"[PS5Debug] pygame unavailable: {type(e).__name__}: {e}")
                return 1
            print(f"[PS5Debug] pygame unavailable, falling back to XInput: {type(e).__name__}")

    return run_xinput(args.controller, args.poll_hz, args.print_all, args.deadband)


if __name__ == "__main__":
    sys.exit(main())
