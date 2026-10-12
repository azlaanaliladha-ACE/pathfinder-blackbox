"""Shared look for every panel: black background, neon-blue headers and readouts.

Palette is BGR.
"""
import cv2
import numpy as np

import config

WHITE = (255, 248, 240)
GREY = (150, 115, 90)
NEON = (255, 170, 0)                 # neon blue
ICE = (255, 235, 150)                # pale cyan readouts
DIM = (90, 45, 10)                   # rules and empty gauges
RED = (70, 50, 255)
BG = (6, 3, 0)
HEADER = (30, 14, 2)
FONT = cv2.FONT_HERSHEY_DUPLEX
MONO = cv2.FONT_HERSHEY_PLAIN


def blank() -> np.ndarray:
    w, h = config.PANEL_SIZE
    img = np.empty((h, w, 3), np.uint8)
    img[:] = BG
    return img


def label(img, title):
    """Header bar: amber tag + title in caps."""
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 20), HEADER, -1)
    cv2.rectangle(img, (0, 0), (4, 20), NEON, -1)
    cv2.putText(img, title.upper(), (10, 14), FONT, 0.38, NEON, 1, cv2.LINE_AA)
    cv2.line(img, (0, 20), (w, 20), DIM, 1)


def text(img, s, x, y, colour=WHITE, scale=0.38):
    cv2.putText(img, s, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)


def mono(img, s, x, y, colour=ICE, scale=0.95):
    """Readout digits in a monospace-looking font."""
    cv2.putText(img, s, (x, y), MONO, scale, colour, 1, cv2.LINE_AA)


def legend(img, items):
    x, y = 6, img.shape[0] - 6
    for name, colour in items:
        cv2.rectangle(img, (x, y - 8), (x + 8, y), colour, -1)
        text(img, name.upper(), x + 11, y, GREY, 0.3)
        x += 16 + 7 * len(name)


def finish(img) -> np.ndarray:
    """Scanlines + corner brackets, applied last to every window."""
    h, w = img.shape[:2]
    img[::3] = (img[::3] * 0.82).astype(np.uint8)
    L, c = 14, NEON
    for x, y, dx, dy in ((1, 1, 1, 1), (w - 2, 1, -1, 1), (1, h - 2, 1, -1), (w - 2, h - 2, -1, -1)):
        cv2.line(img, (x, y), (x + dx * L, y), c, 2)
        cv2.line(img, (x, y), (x, y + dy * L), c, 2)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), DIM, 1)
    return img
