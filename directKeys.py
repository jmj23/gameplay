import ctypes
import time
from ctypes import Structure, byref, c_long, windll

import numpy as np
import pyautogui as p

SendInput = ctypes.windll.user32.SendInput

W = 0x11
A = 0x1E
S = 0x1F
D = 0x20
M = 0x32
K = 0x25
SPACE = 0x39

# C struct redefinitions
PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]


class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]


def queryMousePosition():
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    return pt
    # return { "x": pt.x, "y": pt.y}/


def click(x, y):
    # convert to ctypes pixels
    # x = int(x * 0.666)
    # y = int(y * 0.666)
    ctypes.windll.user32.SetCursorPos(x, y)
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # left down
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # left up


def drag(xy1, xy2):
    dist = np.sum((np.array(xy1)-np.array(xy2))**2)**(1/2)
    num = np.round(dist/2).astype(np.int)
    ctypes.windll.user32.SetCursorPos(xy1[0], xy1[1])
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
    print('clicked')
    time.sleep(.5)
    for x, y in zip(np.linspace(xy1[0], xy2[0], num), np.linspace(xy1[1], xy2[1], num)):
        ctypes.windll.user32.SetCursorPos(int(x), int(y))
        print(x, y)
        time.sleep(.1)
    time.sleep(1)
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)



def multi_drag(coords):

    ctypes.windll.user32.SetCursorPos(coords[0][0], coords[0][1])
    ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)
    time.sleep(1)
    for c in coords[1:]:
        ctypes.windll.user32.SetCursorPos(c[0], c[1])
        time.sleep(2)
    ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)


def moveMouseTo(x, y):
    # convert to ctypes pixels
    # x = int(x * 0.666)
    # y = int(y * 0.666)
    print(x, y)
    ctypes.windll.user32.SetCursorPos(x, y)
    # ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # left down
    # ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # left up


def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, hexKeyCode, 0x0008 | 0x0002,
                        0, ctypes.pointer(extra))
    x = Input(ctypes.c_ulong(1), ii_)
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))


def DetectClick(button, numberOfClicks=1):
    '''Waits watchtime seconds. Returns True on click, False otherwise'''
    if button in (1, '1', 'l', 'L', 'left', 'Left', 'LEFT'):
        bnum = 0x01
    elif button in (2, '2', 'r', 'R', 'right', 'Right', 'RIGHT'):
        bnum = 0x02
    numberOfClicks = int(numberOfClicks)
    assert numberOfClicks > 0
    count = 0
    pos = []
    while True:
        if ctypes.windll.user32.GetKeyState(bnum) not in [0, 1]:
            # ^ this returns either 0 or 1 when button is not being held down
            pt = queryMousePosition()
            pos.append([pt.x, pt.y])
            count += 1
            if count == numberOfClicks:
                break
            time.sleep(.2)
        time.sleep(0.01)
    return pos
