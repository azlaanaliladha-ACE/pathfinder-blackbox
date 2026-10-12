"""Top-right view: metric depth map in the neon theme, beside the camera feed.

Near = white/cyan, far = deep blue fading to black. Colours are spread by rank
(histogram-equalised, smoothed over time) so the crowded near range keeps its detail.
Thin iso-depth lines at round distances, and a colour bar labelled in metres.
"""
import cv2
import numpy as np

from terrain_nav.hud import draw_status
from terrain_nav.panels import GREY, NEON, WHITE, text

_STOPS = [(0.0, (255, 255, 255)), (0.18, (255, 240, 90)), (0.45, (255, 150, 0)),
          (0.75, (130, 25, 0)), (1.0, (12, 2, 0))]


def _lut() -> np.ndarray:
    t = np.linspace(0, 1, 256)
    out = np.zeros((256, 3))
    for (t0, c0), (t1, c1) in zip(_STOPS, _STOPS[1:]):
        sel = (t >= t0) & (t <= t1)
        k = ((t[sel] - t0) / (t1 - t0))[:, None]
        out[sel] = (1 - k) * np.array(c0) + k * np.array(c1)
    return out.astype(np.uint8)


LUT = _lut()
QUANTS = np.linspace(0, 100, 21)


class DepthView:
    def __init__(self):
        self.q = None

    def render(self, depth: np.ndarray, title_right: str = "") -> np.ndarray:
        h, w = depth.shape
        q = np.percentile(depth[::4, ::4], QUANTS)
        self.q = q if self.q is None else 0.85 * self.q + 0.15 * q
        self.q = np.maximum.accumulate(self.q + np.arange(len(QUANTS)) * 1e-3)   # strictly increasing
        lo, hi = self.q[0], self.q[-1]
        norm = lambda d: np.interp(d, self.q, QUANTS / 100.0)
        img = LUT[(norm(depth) * 255).astype(np.uint8)]

        step = _iso_step(hi - lo)
        levels = (depth // step).astype(np.uint8)
        iso = cv2.Canny(levels * 20, 10, 20) > 0
        img[iso] = (0.4 * img[iso] + 0.6 * np.array(NEON)).astype(np.uint8)

        # colour bar
        bx, by0, by1 = w - 28, 44, h - 30
        ramp = LUT[(np.linspace(0, 1, by1 - by0) * 255).astype(np.uint8)]
        img[by0:by1, bx:bx + 10] = ramp[:, None, :]
        cv2.rectangle(img, (bx - 1, by0 - 1), (bx + 10, by1), NEON, 1)
        first = np.ceil(lo / step) * step
        last_y = -100
        for m in np.arange(first, hi, step):
            y = by0 + int((by1 - by0 - 1) * norm(m))
            if y - last_y < 16:                              # far range is compressed: skip crowded ticks
                continue
            last_y = y
            cv2.line(img, (bx - 4, y), (bx - 1, y), WHITE, 1)
            text(img, f"{m:g}", bx - 30, y + 4, WHITE, 0.32)
        text(img, "M", bx + 1, by1 + 14, GREY, 0.32)

        draw_status(img, "DEPTH MAP  //  DEPTH ANYTHING V2", title_right)
        return img


def _iso_step(span: float) -> float:
    for s in (1, 2, 5, 10, 20):
        if span / s <= 8:
            return float(s)
    return 50.0
