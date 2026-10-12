"""Window 1: bird's-eye topographic map built from metric depth.

Points are levelled onto the fitted ground plane, rasterised top-down, coloured by
elevation with contour lines, and marked with water, rocks/rubble, obstacles and the path.
"""
import cv2
import numpy as np

import config
from terrain_nav.boxes3d import Box3D
from terrain_nav.geometry import GroundPlane, backproject
from terrain_nav.panels import WHITE, label, legend

WATER, ROCK, PATH = (230, 140, 30), (60, 90, 150), (255, 0, 255)


def top_view(depth, u, v, pts, plane: GroundPlane, terrain_idx, labels,
             boxes: list[Box3D], path) -> np.ndarray:
    pw, ph = config.PANEL_SIZE
    h, w = depth.shape
    rows, cols = terrain_idx.shape
    elev = plane.elevation(pts)
    flat = plane.level(pts)                                # (lateral, forward) on the ground

    # map extent fitted to the data (rover at bottom-centre), equal metres per pixel on both axes
    z_max = float(np.percentile(flat[:, 1], 95))
    x_half = float(np.percentile(np.abs(flat[:, 0]), 95))
    top, bottom = 22, 18                                   # room for title and legend
    m_per_px = max(2 * x_half / (pw - 8), z_max / (ph - top - bottom))

    def to_px(x, z):
        return (x / m_per_px + pw / 2).astype(int), (ph - bottom - z / m_per_px).astype(int)

    px, py = to_px(flat[:, 0], flat[:, 1])
    ok = (px >= 0) & (px < pw) & (py >= top) & (py < ph - bottom)

    # rasterise mean elevation at half resolution (fills gaps between samples), then upscale
    hx, hy = px[ok] // 2, py[ok] // 2
    acc, cnt = np.zeros((ph // 2, pw // 2)), np.zeros((ph // 2, pw // 2))
    np.add.at(acc, (hy, hx), elev[ok])
    np.add.at(cnt, (hy, hx), 1)
    hit = cnt > 0
    mask = cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    height = cv2.inpaint(np.where(hit, acc / np.maximum(cnt, 1), 0).astype(np.float32),
                         ((~hit) & (mask > 0)).astype(np.uint8), 3, cv2.INPAINT_TELEA)
    height = cv2.resize(height, (pw, ph), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (pw, ph), interpolation=cv2.INTER_NEAREST)

    # elevation colours + contour lines
    lo, hi = np.percentile(elev, [5, 95])
    norm = np.clip((height - lo) / max(hi - lo, 1e-3), 0, 1)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    levels = (norm * 8).astype(np.uint8)
    edges = cv2.Canny(levels * 30, 10, 20) > 0
    img[edges] = (img[edges] * 0.4).astype(np.uint8)

    # terrain classes: water and rocks/rubble from the grid labels
    cell = np.minimum(v * rows // h, rows - 1), np.minimum(u * cols // w, cols - 1)
    point_label = np.array(labels)[terrain_idx[cell]]
    for name, colour in (("water", WATER), ("rubble", ROCK)):
        sel = ok & (point_label == name)
        layer = np.zeros((ph, pw), np.uint8)
        layer[py[sel], px[sel]] = 255
        layer = cv2.dilate(layer, np.ones((5, 5), np.uint8)) > 0
        img[layer] = (0.45 * img[layer] + 0.55 * np.array(colour)).astype(np.uint8)

    img[mask == 0] = 18                                    # no data

    # obstacles as footprints, path, rover
    for b in boxes:
        c = plane.level(b.corners_3d)
        fx, fz = to_px(c[:, 0], c[:, 1])
        cv2.rectangle(img, (fx.min(), fz.min()), (fx.max(), fz.max()), WHITE, 1)
    if path:
        cx = np.array([int((c + 0.5) * w / cols) for _, c in path])
        cy = np.array([int((r + 0.5) * h / rows) for r, _ in path])
        p3 = plane.level(backproject(cx, cy, depth[cy, cx], w, h))
        qx, qz = to_px(p3[:, 0], p3[:, 1])
        cv2.polylines(img, [np.stack([qx, qz], 1).astype(np.int32)], False, PATH, 2, cv2.LINE_AA)
    rx, rz = pw // 2, ph - bottom
    cv2.fillPoly(img, [np.array([[rx, rz - 10], [rx - 6, rz], [rx + 6, rz]], np.int32)], WHITE)

    label(img, f"Top view: topo map  ({z_max:.0f} m ahead)")
    legend(img, [("water", WATER), ("rock", ROCK), ("path", PATH)])
    return img
