"""Persistent world map: remembers every terrain cell the camera has seen, so by the end of a
flight it shows the whole area covered, not just the current frame.

Pose comes from visual odometry in the ground plane: corner features are tracked between
consecutive frames (Lucas-Kanade), lifted to levelled (lateral, forward) ground coordinates
with each frame's metric depth, and a rigid 2D transform is fitted between the two point
sets (RANSAC). Because translation is measured in the same metres as the depth, the map
stays geometrically consistent with what each frame sees. It still drifts over long runs.

Coordinates: metres, world X = right, Z = forward from the start point, height = up from
the ground plane. Each cell keeps a vote count per terrain class (majority wins) and a
height: the highest surface seen in that cell, averaged over the frames that saw it, so
cliffs and ridges keep their tops instead of being averaged into the valley floor.
"""
import math

import cv2
import numpy as np

import config
from terrain_nav.geometry import GroundPlane, backproject
from terrain_nav.panels import BG, GREY, HEADER, NEON, WHITE, blank, label, text

CELL_M = 0.5
EXTENT_M = 600.0                      # map is EXTENT_M x EXTENT_M, start point at the centre
MAX_STEP_M = 1.5                      # reject odometry jumps larger than this per frame
MAX_TURN_DEG = 8.0                    # ...and turns sharper than this per frame
TRAIL_COLOUR = WHITE

# map colour per terrain label (BGR), all in the black / neon-blue theme
CLASS_COLOURS = {
    "road": (170, 160, 150), "dirt": (150, 95, 55), "grass": (215, 140, 20),
    "gravel": (160, 120, 85), "sand": (245, 215, 170), "mud": (110, 45, 25),
    "rubble": (150, 70, 100), "water": (255, 235, 60), "indoor": (130, 130, 130),
}


def _rigid(theta: float, tx: float, tz: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, tx], [s, c, tz], [0, 0, 1.0]])


class GroundOdometry:
    """Frame-to-frame camera motion on the ground plane, as a 3x3 world_from_camera transform.

    Both frames are levelled with the same (current, smoothed) ground plane so plane jitter
    cannot show up as fake rotation, and accepted steps are smoothed because a carried or
    flown camera cannot change velocity much in one frame."""

    SMOOTH = 0.5                       # weight of the new step

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None
        self.T = np.eye(3)
        self.vel = np.zeros(3)         # (theta, tx, tz) per frame; also the fallback when a fit fails
        self.inliers = 0

    def update(self, gray: np.ndarray, depth: np.ndarray, plane: GroundPlane) -> np.ndarray:
        if self.prev is not None:
            step = self._estimate(*self.prev, gray, depth, plane)
            if step is not None:
                self.vel = (1 - self.SMOOTH) * self.vel + self.SMOOTH * np.array(step)
            self.T = self.T @ _rigid(*self.vel)
        self.prev = (gray, depth)
        return self.T

    def _estimate(self, g0, d0, g1, d1, plane):
        h, w = g0.shape
        pts0 = cv2.goodFeaturesToTrack(g0, 600, 0.01, 7)
        if pts0 is None or len(pts0) < 12:
            return None
        pts1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, winSize=(21, 21), maxLevel=3)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, pts1, None, winSize=(21, 21), maxLevel=3)
        good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm(back - pts0, axis=2)[:, 0] < 1.0)
        a, b = pts0[good, 0], pts1[good, 0]
        inside = (b[:, 0] >= 0) & (b[:, 0] < w) & (b[:, 1] >= 0) & (b[:, 1] < h)
        a, b = a[inside], b[inside]
        if len(a) < 12:
            return None
        u0, v0 = a[:, 0].astype(int), a[:, 1].astype(int)
        u1, v1 = b[:, 0].astype(int), b[:, 1].astype(int)
        z0, z1 = d0[v0, u0], d1[v1, u1]
        near = (z0 < config.ODOM_MAX_DEPTH_M) & (z1 < config.ODOM_MAX_DEPTH_M)   # far depth is too noisy
        if near.sum() < 12:
            return None
        A = plane.level(backproject(a[near, 0], a[near, 1], z0[near], w, h))
        B = plane.level(backproject(b[near, 0], b[near, 1], z1[near], w, h))
        thr = max(0.1, 0.05 * float(np.median(z1[near])))  # tolerance scales with scene depth
        M, inl = cv2.estimateAffinePartial2D(B.astype(np.float32), A.astype(np.float32),
                                             method=cv2.RANSAC, ransacReprojThreshold=thr)
        if M is None or inl.sum() < 10:
            return None
        self.inliers = int(inl.sum())
        theta = math.atan2(M[1, 0], M[0, 0])
        tx, tz = M[0, 2], M[1, 2]
        if math.hypot(tx, tz) > MAX_STEP_M or abs(theta) > math.radians(MAX_TURN_DEG):
            return None
        return theta, tx, tz


class WorldMap:
    def __init__(self, labels: list[str]):
        self.labels = labels
        self.colours = np.array([CLASS_COLOURS.get(l, (110, 110, 110)) for l in labels], np.uint8)
        self.n = int(EXTENT_M / CELL_M)
        self.votes = np.zeros((self.n, self.n, len(labels)), np.uint16)
        self.h_sum = np.zeros(self.n * self.n, np.float32)
        self.h_cnt = np.zeros(self.n * self.n, np.float32)
        self.trail: list[tuple[float, float, float]] = []   # (x, z, altitude) per frame
        self.footprint = np.empty((0, 2))
        self.T = np.eye(3)

    def reset(self):
        self.votes[:] = 0
        self.h_sum[:] = 0
        self.h_cnt[:] = 0
        self.trail.clear()
        self.footprint = np.empty((0, 2))
        self.T = np.eye(3)

    def add_frame(self, flat, class_idx, elev, T, altitude: float):
        """flat: [N,2] levelled (lateral, forward) points; class_idx, elev: [N];
        T: world_from_camera; altitude: camera height above the ground plane."""
        self.T = T
        world = flat @ T[:2, :2].T + T[:2, 2]
        self.trail.append((T[0, 2], T[1, 2], altitude))
        ix = np.floor(world[:, 0] / CELL_M).astype(int) + self.n // 2
        iz = np.floor(world[:, 1] / CELL_M).astype(int) + self.n // 2
        ok = (ix >= 0) & (ix < self.n) & (iz >= 0) & (iz < self.n)
        if not ok.any():
            return
        ix, iz, elev = ix[ok], iz[ok], elev[ok].astype(np.float32)
        np.add.at(self.votes, (iz, ix, class_idx[ok]), 1)

        # highest surface per cell in this frame, then averaged across frames
        lin = iz * self.n + ix
        order = np.argsort(lin, kind="stable")
        cells, start = np.unique(lin[order], return_index=True)
        self.h_sum[cells] += np.maximum.reduceat(elev[order], start)
        self.h_cnt[cells] += 1

        ground = np.abs(elev) < 1.0                          # footprint = ground the camera sees now
        if ground.sum() >= 3:
            self.footprint = cv2.convexHull(world[ok][ground].astype(np.float32))[:, 0]

    @property
    def area_m2(self) -> float:
        return float((self.h_cnt > 0).sum()) * CELL_M * CELL_M

    @property
    def trail_length_m(self) -> float:
        if len(self.trail) < 2:
            return 0.0
        t = np.array(self.trail)[:, :2]
        return float(np.linalg.norm(np.diff(t, axis=0), axis=1).sum())

    def surface(self, pad: int = 4):
        """Crop of everything seen: colours, heights (m), mask, class index and crop origin.
        Rows run with world Z (row 0 = smallest Z). Holes between samples are filled."""
        seen = (self.h_cnt > 0).reshape(self.n, self.n)
        if not seen.any():
            return None
        rows, cols = np.where(seen)
        r0, r1 = max(rows.min() - pad, 0), min(rows.max() + 1 + pad, self.n)
        c0, c1 = max(cols.min() - pad, 0), min(cols.max() + 1 + pad, self.n)
        sub = seen[r0:r1, c0:c1].astype(np.uint8)
        cls = np.argmax(self.votes[r0:r1, c0:c1], axis=2)
        hs = self.h_sum.reshape(self.n, self.n)[r0:r1, c0:c1]
        hc = self.h_cnt.reshape(self.n, self.n)[r0:r1, c0:c1]
        height = np.where(sub > 0, hs / np.maximum(hc, 1), 0).astype(np.float32)

        mask = cv2.morphologyEx(sub, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        holes = ((mask > 0) & (sub == 0)).astype(np.uint8)
        colour = cv2.inpaint(self.colours[cls], holes, 2, cv2.INPAINT_TELEA)
        height = cv2.inpaint(height, holes, 2, cv2.INPAINT_TELEA)
        height = cv2.GaussianBlur(cv2.medianBlur(height, 5), (0, 0), 2.0)   # keep cliffs, drop spikes
        return dict(colour=colour, height=height, mask=mask > 0, cls=cls, seen=sub > 0,
                    r0=r0, r1=r1, c0=c0, c1=c1)

    def to_cell(self, x, z, s):
        """World metres -> (col, row) inside a surface() crop."""
        return (np.asarray(x) / CELL_M + self.n // 2 - s["c0"],
                np.asarray(z) / CELL_M + self.n // 2 - s["r0"])

    def legend_items(self, s, k: int = 5):
        counts = np.bincount(s["cls"][s["seen"]].ravel(), minlength=len(self.labels))
        order = [i for i in np.argsort(-counts) if counts[i] > 0][:k]
        total = max(counts.sum(), 1)
        return [(f"{self.labels[i]} {counts[i] / total:.0%}", tuple(int(c) for c in self.colours[i])) for i in order]

    def render(self, size=None, title="World map: whole flight") -> np.ndarray:
        """Everything seen so far, top-down, forward = up. Hill-shaded with height contours."""
        pw, ph = size or config.PANEL_SIZE
        big = size is not None
        img = blank() if not big else np.full((ph, pw, 3), BG, np.uint8)
        top, bottom = (22, 22) if not big else (70, 50)
        fs = 1.0 if not big else 2.0                         # text/line scale factor

        s = self.surface()
        if s is None:
            label(img, title)
            text(img, "BUILDING MAP...", pw // 2 - 50, ph // 2, GREY, 0.42)
            return img

        colour, height, mask = s["colour"], s["height"], s["mask"].astype(np.uint8)
        gz, gx = np.gradient(height / CELL_M)
        shade = np.clip(0.75 + 0.6 * (-gx - gz) / np.sqrt(1 + gx ** 2 + gz ** 2), 0.4, 1.3)
        colour = np.clip(colour * shade[..., None], 0, 255).astype(np.uint8)

        # world Z grows upward on screen
        colour, mask, height = colour[::-1], mask[::-1], height[::-1]
        span_r, span_c = s["r1"] - s["r0"], s["c1"] - s["c0"]
        scale = min((pw - 16) / span_c, (ph - top - bottom) / span_r)
        out_w, out_h = max(int(span_c * scale), 1), max(int(span_r * scale), 1)
        ox, oy = (pw - out_w) // 2, top + ((ph - top - bottom) - out_h) // 2
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        colour = cv2.resize(colour, (out_w, out_h), interpolation=interp)
        mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
        height = cv2.resize(height, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        lo, hi = np.percentile(height[mask], [5, 98])
        levels = np.clip((height - lo) / max(hi - lo, 1e-3) * 10, 0, 10).astype(np.uint8)
        contour = (cv2.Canny(levels * 25, 10, 20) > 0) & mask
        colour[contour] = (0.5 * colour[contour] + 0.5 * np.array(NEON)).astype(np.uint8)
        roi = img[oy:oy + out_h, ox:ox + out_w]
        roi[mask] = colour[mask]

        def to_px(x, z):
            c, r = self.to_cell(x, z, s)
            return c * scale + ox, (span_r - r) * scale + oy

        th = max(1, int(round(fs)))
        if len(self.footprint) >= 3:                         # what the camera sees right now
            fx, fz = to_px(self.footprint[:, 0], self.footprint[:, 1])
            cv2.polylines(img, [np.stack([fx, fz], 1).astype(np.int32)], True, NEON, th, cv2.LINE_AA)
        if len(self.trail) > 1:
            tr = np.array(self.trail)
            tx, tz = to_px(tr[:, 0], tr[:, 1])
            cv2.polylines(img, [np.stack([tx, tz], 1).astype(np.int32)], False, TRAIL_COLOUR, 2 * th, cv2.LINE_AA)
            cv2.circle(img, (int(tx[0]), int(tz[0])), int(4 * fs), TRAIL_COLOUR, -1, cv2.LINE_AA)

        # camera marker: arrow along its forward axis
        cx, cz = to_px(self.T[0, 2], self.T[1, 2])
        fwd = self.T[:2, 1]
        L = 12 * fs
        tip = (int(cx + fwd[0] * L), int(cz - fwd[1] * L))
        cv2.arrowedLine(img, (int(cx), int(cz)), tip, NEON, 2 * th, cv2.LINE_AA, tipLength=0.45)
        cv2.circle(img, (int(cx), int(cz)), int(4 * fs), WHITE, -1, cv2.LINE_AA)

        # scale bar
        bar_m = nice(0.25 * (pw - 16) / scale * CELL_M)
        bar_px = int(bar_m / CELL_M * scale)
        bx, by = pw - bar_px - int(10 * fs), ph - bottom + int(14 * fs)
        cv2.line(img, (bx, by), (bx + bar_px, by), WHITE, th)
        for x in (bx, bx + bar_px):
            cv2.line(img, (x, by - int(3 * fs)), (x, by + int(3 * fs)), WHITE, th)
        text(img, f"{bar_m:g} m", bx, by - int(6 * fs), WHITE, 0.3 * fs)

        x, y = int(6 * fs), ph - int(6 * fs)
        for name, c in self.legend_items(s, 5 if big else 3):
            cv2.rectangle(img, (x, y - int(8 * fs)), (x + int(8 * fs), y), c, -1)
            text(img, name.upper(), x + int(11 * fs), y, GREY, 0.3 * fs)
            x += int((16 + 6.5 * len(name)) * fs)

        if big:
            title_bar(img, title, f"{self.area_m2:,.0f} m2 MAPPED   {self.trail_length_m:,.0f} m FLOWN")
        else:
            label(img, title)
            text(img, f"{self.area_m2:,.0f} m2", pw - 70, 14, WHITE, 0.32)
        return img


def title_bar(img, title: str, right: str = ""):
    """Large header for full-frame cards."""
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 52), HEADER, -1)
    cv2.rectangle(img, (0, 0), (8, 52), NEON, -1)
    cv2.line(img, (0, 52), (w, 52), NEON, 1)
    text(img, title.upper(), 22, 35, NEON, 0.9)
    if right:
        (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
        text(img, right, w - tw - 20, 33, WHITE, 0.6)


def nice(x: float) -> float:
    """Round down to 1, 2 or 5 x 10^k for a readable scale bar."""
    if x <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(x))
    return max(m for m in (1, 2, 5, 10) if m * p <= x) * p
