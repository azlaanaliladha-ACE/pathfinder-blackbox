"""Window 4: terrain cost map.

Each grid cell is coloured by its traversal cost (the same colours as the main feed),
the planned path is drawn over it, and the bottom bars show how much of the visible
ground belongs to each terrain class.
"""
import cv2
import numpy as np

import config
from terrain_nav.hud import cost_colour
from terrain_nav.panels import NEON, GREY, WHITE, blank, label, text

HORIZON_FILL = (24, 10, 2)
CELL_TOP, CELL_H, CELL_LEFT = 30, 120, 8
BAR_TOP, BAR_STEP, MAX_BARS = 166, 14, 5


def costmap_panel(idx, labels, cost, path) -> np.ndarray:
    img = blank()
    label(img, "Terrain stats // cost map")
    pw, _ = config.PANEL_SIZE
    rows, cols = cost.shape
    cw = (pw - 2 * CELL_LEFT) / cols
    ch = CELL_H / rows

    for r in range(rows):
        for c in range(cols):
            x0 = int(CELL_LEFT + c * cw)
            y0 = int(CELL_TOP + r * ch)
            x1 = int(CELL_LEFT + (c + 1) * cw) - 1
            y1 = int(CELL_TOP + (r + 1) * ch) - 1
            colour = HORIZON_FILL if r < config.PLAN_FIRST_ROW else cost_colour(cost[r, c])
            cv2.rectangle(img, (x0, y0), (x1, y1), colour, -1)

    if path:
        pts = np.array([[CELL_LEFT + (c + 0.5) * cw, CELL_TOP + (r + 0.5) * ch] for r, c in path], np.int32)
        cv2.polylines(img, [pts], False, WHITE, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(pts[-1]), 3, WHITE, -1)

    # class coverage over the drivable rows only
    drivable = idx[config.PLAN_FIRST_ROW:]
    counts = np.bincount(drivable.ravel(), minlength=len(labels))
    total = max(counts.sum(), 1)
    order = np.argsort(-counts)[:MAX_BARS]
    bar_w = pw - 2 * CELL_LEFT - 90
    for i, k in enumerate(order):
        share = counts[k] / total
        y = BAR_TOP + i * BAR_STEP
        text(img, labels[k].upper(), CELL_LEFT, y, GREY, 0.3)
        cv2.rectangle(img, (CELL_LEFT + 74, y - 8), (CELL_LEFT + 74 + int(bar_w * share), y), NEON, -1)
        text(img, f"{share:.0%}", pw - 40, y, WHITE, 0.3)
    return img
