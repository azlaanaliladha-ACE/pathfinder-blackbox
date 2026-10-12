"""Bottom panel 3 and camera banner for the Nemotron copilot."""
import textwrap

import cv2
import numpy as np

import config
from terrain_nav.copilot import Copilot
from terrain_nav.drone import DroneTelemetry
from terrain_nav.panels import DIM, GREY, HEADER, ICE, NEON, RED, WHITE, blank, label, mono, text

CALM = ("PROCEED", "STANDBY")


def action_colour(action: str, risk: float):
    if action in ("HOLD", "RETURN") or risk >= 0.7:
        return RED
    return ICE if action in CALM and risk < 0.4 else NEON


def copilot_panel(cp: Copilot, drone: DroneTelemetry) -> np.ndarray:
    img = blank()
    label(img, "AI copilot // Monad black box")
    pw = img.shape[1]
    d = cp.last

    # model + link status
    model = (cp.model or "nemotron").split("/")[-1][:30]
    text(img, model.upper(), 8, 36, GREY, 0.3)
    ok = cp.enabled and cp.status in ("ONLINE", "READY")
    cv2.circle(img, (pw - 72, 32), 4, ICE if ok else RED, -1, cv2.LINE_AA)
    text(img, cp.status if not ok else "LINK OK", pw - 64, 36, ICE if ok else RED, 0.3)

    # decision
    colour = action_colour(d.action, d.risk)
    cv2.rectangle(img, (8, 46), (pw - 8, 80), colour, 1)
    text(img, d.action.replace("_", " "), 16, 70, colour, 0.7)
    mono(img, f"RISK {d.risk:0.2f}", pw - 92, 60, colour, 0.9)
    mono(img, f"{d.latency_ms:5.0f}ms", pw - 92, 75, GREY, 0.85)

    # risk bar
    cv2.rectangle(img, (8, 86), (pw - 8, 92), DIM, -1)
    cv2.rectangle(img, (8, 86), (8 + int((pw - 16) * d.risk), 92), colour, -1)

    # reason, wrapped
    for i, line in enumerate(textwrap.wrap(d.reason or "awaiting first decision", 44)[:2]):
        text(img, line, 8, 108 + i * 15, WHITE, 0.36)

    # safety guard verdict and mission objective
    if d.override:
        text(img, f"GUARD: {d.proposed.replace('_', ' ')} -> {d.action.replace('_', ' ')}", 8, 144, RED, 0.34)
    else:
        text(img, "GUARD: PASS" if d.proposed else "GUARD: ARMED", 8, 144, GREY, 0.34)
    obj = (cp.mission or {}).get("objective", "free flight (no mission brief)")
    text(img, ("MISSION: " + obj)[:46], 8, 159, ICE, 0.32)

    # compact state the model was given
    s = drone.s
    cv2.line(img, (8, 166), (pw - 8, 166), DIM, 1)
    for i, (k, v) in enumerate((("ALT", f"{s['alt']:.1f}M"), ("SPD", f"{s['speed']:.1f}M/S"),
                                ("V/S", f"{s['climb']:+.1f}"))):
        x = 8 + i * 104
        text(img, k, x, 184, GREY, 0.32)
        mono(img, v, x + 30, 185, ICE, 0.9)

    cv2.line(img, (8, 196), (pw - 8, 196), DIM, 1)
    text(img, "CALLS", 8, 214, GREY, 0.32)
    mono(img, f"{cp.calls:4d}", 46, 215, ICE, 0.9)
    text(img, "OVERRIDES", 96, 214, GREY, 0.32)
    mono(img, f"{cp.overrides:3d}", 166, 215, RED if cp.overrides else ICE, 0.9)
    ch = cp.chain
    text(img, "ON-CHAIN", 206, 214, GREY, 0.32)
    mono(img, f"{ch.confirmed:4d}", 264, 215, ICE if ch.enabled else RED, 0.9)
    text(img, f"MONAD {ch.status}"[:40] if ch.enabled else "MONAD RECORDER OFF (SET KEY)", 8, 232,
         ICE if ch.enabled and not ch.failed else RED, 0.28)
    return img


def draw_callout(frame, cp: Copilot):
    """Nemotron's spoken-style callout across the bottom of the camera feed."""
    d = cp.last
    if not cp.enabled or not d.callout:
        return
    h, w = frame.shape[:2]
    msg = f"{'GUARD' if d.override else 'COPILOT'}: {d.callout.upper()}"
    (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
    x0, y0 = (w - tw) // 2 - 12, h - 40
    colour = action_colour(d.action, d.risk)
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 24, y0 + 28), HEADER, -1)
    cv2.rectangle(frame, (x0, y0), (x0 + tw + 24, y0 + 28), colour, 1)
    cv2.putText(frame, msg, (x0 + 12, y0 + 19), cv2.FONT_HERSHEY_DUPLEX, 0.55, colour, 1, cv2.LINE_AA)
