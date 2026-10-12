"""End-of-flight 3D view: the whole world map as a lit height surface, orbited 360 degrees.

Pure OpenCV software renderer (no GL): the height grid becomes a quad mesh, quads are
shaded by terrain class, height and a fixed sun, then drawn back-to-front (painter's
algorithm). The drone's flight path is drawn at its estimated altitude above the
ground plane, with drop lines to the surface so the height of the flight reads clearly.
"""
import math

import cv2
import numpy as np

from terrain_nav.panels import BG, DIM, GREY, ICE, NEON, WHITE, text
from terrain_nav.world_map import CELL_M, WorldMap, nice, title_bar

MAX_CELLS = 170                       # mesh resolution cap on the long side
WIRE_EVERY = 6                        # neon wireframe on every Nth row/column
DROP_EVERY = 25                       # altitude drop line every N trail points
SUN = np.array([-0.5, 0.8, -0.35])
LOW, HIGH = np.array([70, 15, 0.0]), np.array([255, 250, 200.0])   # valley -> ridge tint (BGR)

# "clean" style: plain white studio look with hypsometric (topographic) colouring by height
CLEAN_BG = (246, 246, 244)
CLEAN_GRID = (222, 222, 218)
CLEAN_TRAIL = (20, 20, 20)
_TOPO = [(0.0, (60, 120, 70)), (0.25, (95, 160, 110)), (0.45, (150, 200, 220)),
         (0.65, (90, 140, 185)), (0.85, (70, 100, 140)), (1.0, (235, 238, 240))]   # BGR green->tan->brown->snow


def topo_colour(t: np.ndarray) -> np.ndarray:
    """Hypsometric tint (BGR float) for heights normalised to [0, 1]."""
    t = np.clip(t, 0, 1)
    out = np.zeros(t.shape + (3,))
    for (t0, c0), (t1, c1) in zip(_TOPO, _TOPO[1:]):
        sel = (t >= t0) & (t <= t1)
        k = ((t[sel] - t0) / (t1 - t0))[:, None]
        out[sel] = (1 - k) * np.array(c0) + k * np.array(c1)
    return out


class Orbit:
    def __init__(self, world: WorldMap, size, style: str = "hud", zoom: float = 1.0, max_cells: int = MAX_CELLS):
        """style: "hud" (neon, with readouts), "clean" (white background, topo colours, no text)
        or "photo" (white background, the map's own colours, i.e. a textured model).
        zoom > 1 fills more of the frame (parts may leave it at some angles)."""
        self.W, self.H = size
        self.style = style
        self.top = 52 if style == "hud" else 0               # room for the title bar
        s = world.surface()
        if s is None:
            raise ValueError("world map is empty")
        rows, cols = s["height"].shape
        f = max(1, math.ceil(max(rows, cols) / max_cells))
        gw, gh = cols // f, rows // f
        colour = cv2.resize(s["colour"], (gw, gh), interpolation=cv2.INTER_AREA).astype(float)
        height = cv2.resize(s["height"], (gw, gh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(s["mask"].astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0
        cell = CELL_M * f

        lo, hi = np.percentile(height[mask], [2, 99])
        span = max(gw, gh) * cell
        self.exag = float(np.clip(0.25 * span / max(hi - lo, 0.5), 1.0, 4.0))
        self.lo, self.hi = float(lo), float(hi)
        cx, cz = gw / 2, gh / 2

        X = (np.arange(gw)[None, :] - cx) * cell * np.ones((gh, 1))
        Z = (np.arange(gh)[:, None] - cz) * cell * np.ones((1, gw))
        Y = (np.clip(height, lo, hi) - lo) * self.exag
        self.verts = np.stack([X, Y, Z], -1)                 # [gh, gw, 3]
        self.base_y = 0.0

        # quads whose four corners are all mapped
        ok = mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]
        r, c = np.where(ok)
        self.quads = np.stack([r * gw + c, r * gw + c + 1, (r + 1) * gw + c + 1, (r + 1) * gw + c], 1)
        self.wire = (r % WIRE_EVERY == 0) | (c % WIRE_EVERY == 0)

        # per-quad colour: class colour blended with a height tint, lit by the sun
        V = self.verts.reshape(-1, 3)
        p0, p1, p3 = V[self.quads[:, 0]], V[self.quads[:, 1]], V[self.quads[:, 3]]
        n = np.cross(p3 - p0, p1 - p0)
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
        n[n[:, 1] < 0] *= -1
        lit = np.clip(0.35 + 0.85 * n @ (SUN / np.linalg.norm(SUN)), 0.25, 1.2)
        hq = (height[r, c] - lo) / max(hi - lo, 1e-3)
        tint = LOW + np.clip(hq, 0, 1)[:, None] * (HIGH - LOW)
        if style == "photo":
            self.qcol = np.clip(colour[r, c] * (0.7 + 0.3 * lit)[:, None], 0, 255).astype(np.uint8)
            self.wire[:] = False
        else:
            base = 0.3 * colour[r, c] + 0.7 * tint if style == "hud" else topo_colour(hq)
            self.qcol = np.clip(base * lit[:, None], 0, 255).astype(np.uint8)
        self.qwire = (self.qcol * (0.82 if style == "clean" else 0)).astype(np.uint8)

        # drone trail at altitude (same exaggeration as the terrain so heights compare)
        tr = np.array(world.trail) if world.trail else np.zeros((0, 3))
        tc, trr = world.to_cell(tr[:, 0], tr[:, 1], s) if len(tr) else (np.zeros(0), np.zeros(0))
        tx, tz = (tc / f - cx) * cell, (trr / f - cz) * cell
        ground = self._ground_at(height, tc / f, trr / f, lo)
        alt = np.clip(tr[:, 2], 0, None) if len(tr) else np.zeros(0)
        self.trail = np.stack([tx, (ground + alt - lo) * self.exag, tz], 1) if len(tr) else np.zeros((0, 3))
        self.trail_ground = np.stack([tx, (ground - lo) * self.exag, tz], 1) if len(tr) else np.zeros((0, 3))

        self.centre = np.array([0.0, 0.35 * (hi - lo) * self.exag, 0.0])
        self.radius = 1.15 * span
        self.span = span
        self.f = self._fit_focal() * zoom

    @staticmethod
    def _ground_at(height, c, r, lo):
        if len(c) == 0:
            return np.zeros(0)
        gh, gw = height.shape
        ci = np.clip(np.round(c).astype(int), 0, gw - 1)
        ri = np.clip(np.round(r).astype(int), 0, gh - 1)
        return np.maximum(height[ri, ci], lo)

    def _camera(self, az, el):
        eye = self.centre + self.radius * np.array([math.cos(el) * math.sin(az), math.sin(el),
                                                    -math.cos(el) * math.cos(az)])
        fwd = self.centre - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross([0, 1, 0], fwd)                   # world X stays on screen right (no mirror)
        right /= np.linalg.norm(right)
        up = np.cross(fwd, right)
        return eye, right, up, fwd

    def _project(self, P, cam, f):
        eye, right, up, fwd = cam
        d = P - eye
        x, y, z = d @ right, d @ up, np.maximum(d @ fwd, 1e-3)
        u = self.W / 2 + f * x / z
        v = self.top + (self.H - self.top) / 2 - f * y / z
        return np.stack([u, v], -1), z

    def _fit_focal(self):
        V = self.verts.reshape(-1, 3)[np.unique(self.quads)]
        worst = 0.0
        for az in np.linspace(0, 2 * math.pi, 12, endpoint=False):
            for el in (math.radians(25), math.radians(60)):
                uv, _ = self._project(V, self._camera(az, el), 1.0)
                worst = max(worst, np.abs(uv[:, 0] - self.W / 2).max() / (0.44 * self.W),
                            np.abs(uv[:, 1] - (self.top + (self.H - self.top) / 2)).max() / (0.42 * (self.H - self.top)))
        return 1.0 / max(worst, 1e-6)

    def frame(self, t: float, title: str) -> np.ndarray:
        """t in [0, 1): one full turn of azimuth; elevation swings between 25 and 60 degrees."""
        az = 2 * math.pi * t
        el = math.radians(42 + 18 * math.sin(2 * math.pi * t))
        cam = self._camera(az, el)
        clean = self.style in ("clean", "photo")
        img = np.full((self.H, self.W, 3), CLEAN_BG if clean else BG, np.uint8)
        self._floor_grid(img, cam)

        V = self.verts.reshape(-1, 3)
        uv, z = self._project(V, cam, self.f)
        quv = uv[self.quads].astype(np.int32)
        qz = z[self.quads].mean(1)

        items = [(qz[i], 0, i) for i in range(len(qz))]
        if len(self.trail) > 1:
            tuv, tz = self._project(self.trail, cam, self.f)
            guv, gz = self._project(self.trail_ground, cam, self.f)
            items += [((tz[i] + tz[i + 1]) / 2, 1, i) for i in range(len(tz) - 1)]
            items += [((tz[i] + gz[i]) / 2, 2, i) for i in range(0, len(tz), DROP_EVERY)]
        items.sort(key=lambda it: -it[0])

        for _, kind, i in items:
            if kind == 0:
                cv2.fillConvexPoly(img, quv[i], tuple(int(c) for c in self.qcol[i]))
                if self.wire[i]:
                    cv2.polylines(img, [quv[i]], True, tuple(int(c) for c in self.qwire[i]) if clean else DIM, 1)
            elif kind == 1:
                cv2.line(img, tuple(tuv[i].astype(int)), tuple(tuv[i + 1].astype(int)),
                         CLEAN_TRAIL if clean else WHITE, 2, cv2.LINE_AA)
            else:
                cv2.line(img, tuple(tuv[i].astype(int)), tuple(guv[i].astype(int)),
                         (150, 150, 150) if clean else NEON, 1, cv2.LINE_AA)

        if clean:
            if len(self.trail) > 1:
                cv2.circle(img, tuple(tuv[-1].astype(int)), 8, CLEAN_TRAIL, -1, cv2.LINE_AA)
                cv2.circle(img, tuple(tuv[-1].astype(int)), 14, CLEAN_TRAIL, 1, cv2.LINE_AA)
            return img
        if len(self.trail) > 1:
            cv2.circle(img, tuple(tuv[0].astype(int)), 6, WHITE, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(tuv[-1].astype(int)), 7, NEON, -1, cv2.LINE_AA)
            cv2.circle(img, tuple(tuv[-1].astype(int)), 11, WHITE, 1, cv2.LINE_AA)
            text(img, "START", int(tuv[0][0]) + 9, int(tuv[0][1]) - 6, WHITE, 0.42)
            text(img, "DRONE", int(tuv[-1][0]) + 13, int(tuv[-1][1]) - 8, NEON, 0.42)

        title_bar(img, title, f"AZ {math.degrees(az) % 360:5.1f}   EL {math.degrees(el):4.1f}   "
                              f"VERTICAL x{self.exag:.1f}")
        self._height_bar(img)
        self._compass(img, az)
        return img

    def _floor_grid(self, img, cam):
        """Reference grid on the ground plane under the map."""
        half = 0.75 * self.span
        step = nice(self.span / 8)
        k = np.arange(-half, half + 1e-6, step)
        for a in k:
            for P in (np.array([[a, 0, -half], [a, 0, half]]), np.array([[-half, 0, a], [half, 0, a]])):
                pts = np.linspace(P[0], P[1], 24)
                uv, z = self._project(pts, cam, self.f)
                cv2.polylines(img, [uv.astype(np.int32)], False,
                              CLEAN_GRID if self.style in ("clean", "photo") else DIM, 1, cv2.LINE_AA)

    def _height_bar(self, img):
        x, y0, y1 = 24, 90, 290
        for y in range(y0, y1):
            k = (y1 - y) / (y1 - y0)
            cv2.line(img, (x, y), (x + 12, y), tuple(int(c) for c in LOW + k * (HIGH - LOW)), 1)
        cv2.rectangle(img, (x - 1, y0 - 1), (x + 13, y1), NEON, 1)
        text(img, "HEIGHT", x - 4, y0 - 10, NEON, 0.42)
        text(img, f"{self.hi - self.lo:+.0f} m", x + 20, y0 + 8, ICE, 0.42)
        text(img, "0 m", x + 20, y1, ICE, 0.42)
        text(img, "drone path at", x - 4, y1 + 26, GREY, 0.38)
        text(img, "flight altitude", x - 4, y1 + 42, GREY, 0.38)

    def _compass(self, img, az):
        cx, cy, r = self.W - 60, self.H - 60, 34
        cv2.circle(img, (cx, cy), r, NEON, 1, cv2.LINE_AA)
        a = -az                                              # world +Z (start heading) on screen
        tip = (int(cx + r * 0.8 * math.sin(a)), int(cy - r * 0.8 * math.cos(a)))
        cv2.arrowedLine(img, (cx, cy), tip, WHITE, 2, cv2.LINE_AA, tipLength=0.3)
        text(img, "FWD", cx - 13, cy + r + 16, GREY, 0.36)
