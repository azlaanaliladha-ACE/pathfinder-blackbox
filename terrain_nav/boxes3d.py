"""Lift 2D obstacle boxes to approximate 3D cuboids using the depth map.

Assumes a pinhole camera and a roughly square footprint (depth extent = width).
Good enough for visualisation and distance estimates, not for precise geometry.
"""
from dataclasses import dataclass

import numpy as np

import config
from terrain_nav.obstacles import Obstacle


@dataclass
class Box3D:
    obstacle: Obstacle
    distance_m: float
    size_m: tuple[float, float, float]   # width, height, depth
    corners_px: np.ndarray               # 8 x 2 projected corners
    corners_3d: np.ndarray               # 8 x 3 corners in camera coordinates (metres)


def _project(pts: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    return np.stack([f * pts[:, 0] / pts[:, 2] + cx, f * pts[:, 1] / pts[:, 2] + cy], axis=1)


def lift(obstacle: Obstacle, depth: np.ndarray) -> Box3D:
    h, w = depth.shape
    f, cx, cy = config.focal_px(w), w / 2, h / 2
    x1, y1, x2, y2 = obstacle.box
    z = float(np.median(depth[max(y1, 0):y2, max(x1, 0):x2]))   # front face distance

    X1, X2 = (x1 - cx) * z / f, (x2 - cx) * z / f
    Y1, Y2 = (y1 - cy) * z / f, (y2 - cy) * z / f
    d = max(X2 - X1, config.MIN_BOX_DEPTH_M)                     # assumed depth extent
    corners = np.array([[X, Y, Z] for Z in (z, z + d) for Y in (Y1, Y2) for X in (X1, X2)])
    return Box3D(obstacle, z, (X2 - X1, Y2 - Y1, d), _project(corners, f, cx, cy), corners)


# Corner index pairs forming the 12 cuboid edges (corner order: Z, Y, X nested)
EDGES = [(0, 1), (2, 3), (0, 2), (1, 3),          # front face
         (4, 5), (6, 7), (4, 6), (5, 7),          # back face
         (0, 4), (1, 5), (2, 6), (3, 7)]          # connecting edges
