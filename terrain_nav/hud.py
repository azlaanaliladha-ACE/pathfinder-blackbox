"""Drawing on the camera feed: terrain overlay, 3D boxes, planned path, status bar.

Black / neon-blue theme. Terrain cost runs from bright cyan (easy) through deep blue to
violet (hard); only impassable cells are red.
"""
import math

import cv2
import numpy as np

from terrain_nav.boxes3d import EDGES, Box3D
from terrain_nav.panels import HEADER, NEON, RED, WHITE

EASY, SLOW, HARD = (255, 230, 0), (255, 110, 20), (200, 30, 140)
BOX = (255, 255, 120)
PATH = (255, 255, 255)
PATH_GLOW = NEON


def cost_colour(cost: float):
    if math.isinf(cost):
        return RED
    return EASY if cost <= 2.0 else SLOW if cost <= 5.0 else HARD


def draw_terrain(frame, labels, idx, cost, first_row: int, alpha=0.28):
    """Tint and label only the drivable band (rows >= first_row)."""
    h, w = frame.shape[:2]
    rows, cols = idx.shape
    ch, cw = h // rows, w // cols
    overlay = frame.copy()
    for r in range(first_row, rows):
        for c in range(cols):
            x, y = c * cw, r * ch
            cv2.rectangle(overlay, (x, y), (x + cw, y + ch), cost_colour(cost[r, c]), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for r in range(first_row, rows + 1):                     # thin neon grid
        cv2.line(frame, (0, r * ch), (w, r * ch), NEON, 1)
    for c in range(cols + 1):
        cv2.line(frame, (c * cw, first_row * ch), (c * cw, h), NEON, 1)
    for r in range(first_row, rows):
        for c in range(cols):
            cv2.putText(frame, labels[idx[r, c]], (c * cw + 3, r * ch + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, WHITE, 1, cv2.LINE_AA)


def draw_box3d(frame, b: Box3D):
    pts = b.corners_px.astype(int)
    for i, j in EDGES:
        cv2.line(frame, tuple(pts[i]), tuple(pts[j]), BOX, 2, cv2.LINE_AA)
    x1, y1 = b.obstacle.box[:2]
    cv2.putText(frame, f"{b.obstacle.label} {b.distance_m:.1f}m", (x1, max(y1 - 6, 34)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX, 2, cv2.LINE_AA)


def draw_path(frame, path, grid_shape):
    h, w = frame.shape[:2]
    rows, cols = grid_shape
    if path is None:
        cv2.putText(frame, "NO SAFE PATH", (w // 2 - 110, h // 2), cv2.FONT_HERSHEY_DUPLEX, 1.0, RED, 2, cv2.LINE_AA)
        return
    pts = np.array([[int((c + 0.5) * w / cols), int((r + 0.5) * h / rows)] for r, c in path])
    pts = np.vstack([[w // 2, h - 1], pts])
    glow = frame.copy()
    cv2.polylines(glow, [pts], False, PATH_GLOW, 10, cv2.LINE_AA)
    cv2.addWeighted(glow, 0.45, frame, 0.55, 0, frame)
    cv2.polylines(frame, [pts], False, PATH, 3, cv2.LINE_AA)
    cv2.circle(frame, tuple(pts[-1]), 7, PATH, -1, cv2.LINE_AA)


def draw_status(frame, title: str, right: str):
    """Header strip on a top-row view."""
    w = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (w, 24), HEADER, -1)
    cv2.rectangle(frame, (0, 0), (5, 24), NEON, -1)
    cv2.line(frame, (0, 24), (w, 24), NEON, 1)
    cv2.putText(frame, title, (12, 17), cv2.FONT_HERSHEY_DUPLEX, 0.48, NEON, 1, cv2.LINE_AA)
    (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_DUPLEX, 0.42, 1)
    cv2.putText(frame, right, (w - tw - 10, 17), cv2.FONT_HERSHEY_DUPLEX, 0.42, WHITE, 1, cv2.LINE_AA)
