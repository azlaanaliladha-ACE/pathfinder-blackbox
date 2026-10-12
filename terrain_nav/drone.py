"""Window 2: drone telemetry SIMULATED from the camera feed.

Treats the footage as if filmed by a drone camera and derives:
  altitude      camera height above the fitted ground plane (metric depth)
  pitch / roll  tilt of that ground plane
  ground speed  mean optical flow scaled by altitude
  yaw rate      rotational component of the flow field
  climb rate    flow divergence (expansion = descending / approaching)
  prop RPM      quad-X motor mixer driven by the above
None of this comes from a real flight controller.
"""
import math

import cv2
import numpy as np

import config
from terrain_nav.geometry import GroundPlane
from terrain_nav.panels import DIM, NEON, ICE, GREY, RED, WHITE, blank, label, mono, text

FLOW_SIZE = (160, 120)


class DroneTelemetry:
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev_gray = None
        self.t = 0.0
        self.s = dict(alt=0.0, pitch=0.0, roll=0.0, speed=0.0, heading=0.0,
                      yaw_rate=0.0, climb=0.0, battery=100.0)

    def _ema(self, key, value, k=0.2):
        self.s[key] = (1 - k) * self.s[key] + k * value

    def update(self, frame_bgr, plane: GroundPlane, dt: float, track_shift=None):
        self.t += dt
        gray = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), FLOW_SIZE)
        alt = plane.camera_height
        self._ema("alt", alt)
        self._ema("pitch", plane.pitch_deg)
        self._ema("roll", plane.roll_deg)

        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fw, fh = FLOW_SIZE
            ys, xs = np.mgrid[0:fh, 0:fw]
            rx, ry = xs - fw / 2, ys - fh / 2
            r2 = rx ** 2 + ry ** 2 + 1e-6
            fx, fy = flow[..., 0], flow[..., 1]

            f_px = config.focal_px(fw)
            if track_shift is not None:
                # measured camera motion from TAPNet ground-point tracks (full-res px -> flow px)
                sx = track_shift[0] * fw / config.FRAME_SIZE[0]
                sy = track_shift[1] * fh / config.FRAME_SIZE[1]
                self._ema("speed", math.hypot(sx, sy) * alt / f_px / dt)
                self._ema("yaw_rate", math.degrees(sx / f_px) / dt)
            else:
                trans = math.hypot(fx.mean(), fy.mean())
                self._ema("speed", trans * alt / f_px / dt)
                rot = float(np.mean((rx * fy - ry * fx) / r2))       # rad per frame
                self._ema("yaw_rate", math.degrees(rot) / dt)
            div = float(np.mean((rx * fx + ry * fy) / r2))           # expansion per frame
            self._ema("climb", -div * alt / dt)
        self.prev_gray = gray

        self.s["heading"] = (self.s["heading"] + self.s["yaw_rate"] * dt) % 360
        self.s["battery"] = max(0.0, 100.0 - self.t * config.DRONE_BATTERY_DRAIN_PER_S)

    @property
    def state(self) -> str:
        s = self.s
        if abs(s["yaw_rate"]) > 15:
            return "TURNING"
        if s["climb"] > 0.3:
            return "CLIMBING"
        if s["climb"] < -0.3:
            return "DESCENDING"
        return "HOVER" if s["speed"] < 0.3 else "CRUISE"

    def motor_rpm(self) -> list[float]:
        """Quad-X mixer. Order: front-left, front-right, rear-left, rear-right."""
        s = self.s
        thrust = 1.0 + 0.15 * np.tanh(s["climb"]) + 0.05 * np.tanh(s["speed"] / 5)
        pitch, roll = np.tanh(s["pitch"] / 20) * 0.1, np.tanh(s["roll"] / 20) * 0.1
        yaw = np.tanh(s["yaw_rate"] / 60) * 0.08
        mix = [thrust - pitch + roll - yaw, thrust - pitch - roll + yaw,
               thrust + pitch + roll + yaw, thrust + pitch - roll - yaw]
        return [config.DRONE_HOVER_RPM * math.sqrt(max(m, 0.05)) for m in mix]


def telemetry_panel(drone: DroneTelemetry) -> np.ndarray:
    img = blank()
    s = drone.s
    label(img, "Drone POV // telemetry")

    # state tag
    colour = ICE if drone.state in ("HOVER", "CRUISE") else NEON
    cv2.rectangle(img, (8, 28), (110, 46), colour, 1)
    text(img, drone.state, 14, 42, colour, 0.42)

    rows = [("ALT", f"{s['alt']:6.1f}", "M"), ("SPD", f"{s['speed']:6.1f}", "M/S"),
            ("V/S", f"{s['climb']:+6.1f}", "M/S"), ("HDG", f"{s['heading']:6.0f}", "DEG"),
            ("YAW", f"{s['yaw_rate']:+6.0f}", "D/S"), ("PIT", f"{s['pitch']:+6.1f}", "DEG"),
            ("ROL", f"{s['roll']:+6.1f}", "DEG")]
    for i, (k, val, unit) in enumerate(rows):
        y = 66 + i * 20
        text(img, k, 8, y, GREY, 0.34)
        mono(img, val, 40, y, ICE)
        text(img, unit, 112, y, GREY, 0.28)
        cv2.line(img, (8, y + 5), (140, y + 5), DIM, 1)

    _battery(img, s["battery"], (8, 212))
    _attitude(img, (236, 74), 40, s["pitch"], s["roll"])
    _motors(img, drone.motor_rpm(), (236, 176))
    return img


def _battery(img, pct, origin):
    x, y = origin
    text(img, "BAT", x, y, GREY, 0.34)
    colour = ICE if pct > 40 else NEON if pct > 15 else RED
    for i in range(10):
        filled = pct >= (i + 1) * 10
        cv2.rectangle(img, (x + 32 + i * 9, y - 9), (x + 38 + i * 9, y), colour if filled else DIM, -1)
    mono(img, f"{pct:3.0f}%", x + 126, y, colour, 0.9)


def _attitude(img, centre, r, pitch, roll):
    """Artificial horizon with pitch-ladder ticks, rotated by roll."""
    cx, cy = centre
    size = 2 * r
    ind = np.empty((size, size, 3), np.uint8)
    shift = int(np.clip(pitch, -30, 30) / 30 * r)
    ind[:] = (90, 40, 5)                                      # sky
    ind[r + shift:] = (12, 6, 0)                              # ground
    cv2.line(ind, (0, r + shift), (size, r + shift), WHITE, 1)
    for k in (-20, -10, 10, 20):                              # ladder ticks every 10 deg
        yk = r + shift - int(k / 30 * r)
        half = 10 if abs(k) == 10 else 16
        cv2.line(ind, (r - half, yk), (r + half, yk), ICE, 1)
    rot = cv2.getRotationMatrix2D((r, r), -roll, 1.0)
    ind = cv2.warpAffine(ind, rot, (size, size), borderMode=cv2.BORDER_REPLICATE)
    circle = np.zeros((size, size), np.uint8)
    cv2.circle(circle, (r, r), r - 1, 255, -1)
    roi = img[cy - r:cy + r, cx - r:cx + r]
    roi[circle > 0] = ind[circle > 0]
    cv2.circle(img, centre, r, NEON, 1, cv2.LINE_AA)
    cv2.line(img, (cx - 18, cy), (cx - 6, cy), NEON, 2)    # fixed aircraft symbol
    cv2.line(img, (cx + 6, cy), (cx + 18, cy), NEON, 2)
    cv2.circle(img, centre, 2, NEON, -1)
    text(img, "ATT", cx - 10, cy - r - 4, GREY, 0.3)


def _motors(img, rpms, centre):
    """Four rotors in X layout; ring colour = load."""
    cx, cy = centre
    offs = [(-30, -22), (30, -22), (-30, 22), (30, 22)]
    for dx, dy in offs:
        cv2.line(img, (cx, cy), (cx + dx, cy + dy), GREY, 2)
    cv2.rectangle(img, (cx - 6, cy - 6), (cx + 6, cy + 6), GREY, -1)
    for (dx, dy), rpm in zip(offs, rpms):
        load = float(np.clip(rpm / (config.DRONE_HOVER_RPM * 1.2), 0, 1))
        colour = ICE if load < 0.85 else NEON if load < 0.95 else RED
        cv2.circle(img, (cx + dx, cy + dy), 12, colour, 1, cv2.LINE_AA)
        cv2.ellipse(img, (cx + dx, cy + dy), (12, 12), -90, 0, int(360 * load), colour, 2, cv2.LINE_AA)
        mono(img, f"{rpm / 1000:.1f}K", cx + dx - 14, cy + dy + (28 if dy > 0 else -17), ICE, 0.8)
    text(img, "PROP RPM", cx - 26, cy + 58, GREY, 0.3)
