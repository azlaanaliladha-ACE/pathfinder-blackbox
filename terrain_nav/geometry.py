"""Camera geometry shared by the panels: back-projection and ground-plane fitting.

Camera coordinates: x right, y down, z forward, metres.
"""
from dataclasses import dataclass

import numpy as np

import config


def backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, w: int, h: int) -> np.ndarray:
    """Pixels (u, v) with depth z -> Nx3 camera-space points (pinhole model)."""
    f = config.focal_px(w)
    return np.stack([(u - w / 2) * z / f, (v - h / 2) * z / f, z], axis=1)


@dataclass
class GroundPlane:
    """y = a*x + b*z + c  (y points down, so the ground is at positive y)."""
    a: float
    b: float
    c: float

    @property
    def camera_height(self) -> float:
        """Perpendicular distance from the camera (origin) to the plane."""
        return abs(self.c) / np.sqrt(self.a ** 2 + 1 + self.b ** 2)

    @property
    def roll_deg(self) -> float:
        return float(np.degrees(np.arctan(self.a)))

    @property
    def pitch_deg(self) -> float:
        return float(np.degrees(np.arctan(self.b)))

    def level(self, pts: np.ndarray) -> np.ndarray:
        """Lateral/forward coordinates on the plane, so a tilted camera gives a true top view."""
        up = -np.array([self.a, -1.0, self.b]) / np.sqrt(self.a ** 2 + 1 + self.b ** 2)
        if up[1] > 0:
            up = -up
        fwd = np.array([0.0, 0.0, 1.0]) - up * up[2]
        fwd /= np.linalg.norm(fwd)
        right = np.cross(up, fwd) * -1                     # x right in camera convention
        return np.stack([pts @ right, pts @ fwd], axis=1)

    def elevation(self, pts: np.ndarray) -> np.ndarray:
        """Height of each point above the plane (positive = up)."""
        ground_y = self.a * pts[:, 0] + self.b * pts[:, 2] + self.c
        return (ground_y - pts[:, 1]) / np.sqrt(self.a ** 2 + 1 + self.b ** 2)


def fit_ground_plane(pts: np.ndarray) -> GroundPlane:
    """Robust least-squares plane: fit, drop the worst 30% (rocks, objects), refit."""
    A = np.c_[pts[:, 0], pts[:, 2], np.ones(len(pts))]
    keep = np.ones(len(pts), bool)
    for _ in range(2):
        coef, *_ = np.linalg.lstsq(A[keep], pts[keep, 1], rcond=None)
        resid = np.abs(A @ coef - pts[:, 1])
        keep = resid <= np.percentile(resid, 70)
    return GroundPlane(*coef)


def _normal(a: float, b: float) -> np.ndarray:
    n = np.array([a, -1.0, b])
    return n / np.linalg.norm(n)


class PlaneFilter:
    """Ground plane that stays on the ground: RANSAC fit constrained to tilt near the previous
    plane (so a wall or doorway filling the lower frame cannot win), then smoothed over time,
    because a carried camera's height and tilt do not jump between frames."""

    ITERS = 150
    FIRST_MAX_DEG = 50.0              # first frame: anything within this of "camera level" is a floor
    MAX_DEG = 12.0                    # later frames: tilt change allowed per frame
    MIN_INLIER_FRAC = 0.12
    MAX_DRIFT_DEG = 30.0              # never further than this from the first frame's floor
    MAX_HEIGHT_CHANGE = 0.25          # a new fit may not move camera height more than this fraction
    SMOOTH = 0.25                     # weight of the new fit

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.coef = None
        self.first = None

    def update(self, pts: np.ndarray) -> GroundPlane:
        if not config.PLANE_CONSTRAINED:                     # outdoor/drone: independent robust fit per frame
            return fit_ground_plane(pts)
        prior = _normal(self.coef[0], self.coef[1]) if self.coef is not None else np.array([0.0, -1.0, 0.0])
        max_deg = self.MAX_DEG if self.coef is not None else self.FIRST_MAX_DEG
        fit = self._ransac(pts, prior, max_deg)
        if fit is not None and self.coef is not None and not self._plausible(fit):
            fit = None                                       # bed, desk or door, not the floor
        if fit is not None:
            self.coef = fit if self.coef is None else (1 - self.SMOOTH) * self.coef + self.SMOOTH * fit
        elif self.coef is None:                              # no floor found yet: plain robust fit
            p = fit_ground_plane(pts)
            self.coef = np.array([p.a, p.b, p.c])
        if self.first is None:
            self.first = _normal(self.coef[0], self.coef[1])
        return GroundPlane(*self.coef)

    def _plausible(self, fit) -> bool:
        h_new, h_old = GroundPlane(*fit).camera_height, GroundPlane(*self.coef).camera_height
        if abs(h_new - h_old) > self.MAX_HEIGHT_CHANGE * h_old:
            return False
        drift = np.degrees(np.arccos(np.clip(_normal(fit[0], fit[1]) @ self.first, -1, 1)))
        return drift <= self.MAX_DRIFT_DEG

    def _ransac(self, pts, prior, max_deg):
        n = len(pts)
        if n < 50:
            return None
        cos_max = np.cos(np.radians(max_deg))
        tol = 0.02 + 0.02 * pts[:, 2]                       # depth noise grows with distance
        best, best_count = None, 0
        idx = self.rng.integers(0, n, size=(self.ITERS, 3))
        for i, j, k in idx:
            nrm = np.cross(pts[j] - pts[i], pts[k] - pts[i])
            norm = np.linalg.norm(nrm)
            if norm < 1e-9:
                continue
            nrm /= norm
            if nrm[1] > 0:
                nrm = -nrm                                   # point "up" (camera y is down)
            if nrm @ prior < cos_max:
                continue
            d = -nrm @ pts[i]
            if d < 0:                                        # camera must be above the plane
                continue
            inl = np.abs(pts @ nrm + d) < tol
            count = int(inl.sum())
            if count > best_count:
                best, best_count = inl, count
        if best is None or best_count < self.MIN_INLIER_FRAC * n:
            return None
        A = np.c_[pts[best, 0], pts[best, 2], np.ones(best_count)]
        coef, *_ = np.linalg.lstsq(A, pts[best, 1], rcond=None)
        return coef


def drivable_points(depth: np.ndarray, stride: int):
    """Sample the drivable band (below the horizon line) -> (u, v, points)."""
    h, w = depth.shape
    top = int(config.PLAN_FIRST_ROW * h / config.GRID_ROWS)
    v, u = np.mgrid[top:h:stride, 0:w:stride]
    u, v = u.ravel(), v.ravel()
    return u, v, backproject(u, v, depth[v, u], w, h)
