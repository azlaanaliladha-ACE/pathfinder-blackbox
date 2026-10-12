"""Bottom panel: the software stack running on each frame, with live per-stage timings."""
import cv2
import torch

from terrain_nav.panels import DIM, GREY, ICE, NEON, WHITE, blank, label, mono, text

# stage key -> (display name, model / method)
STAGES = [
    ("terrain", "TERRAIN", "CLIP ViT-B/32 zero-shot"),
    ("depth", "DEPTH", "Depth Anything V2 metric"),
    ("objects", "OBJECTS", "YOLO-World v2-S"),
    ("tracking", "TRACKING", "BootsTAPIR (TAPNet)"),
    ("odometry", "ODOMETRY", "LK flow + RANSAC"),
    ("map", "WORLD MAP", "height + class grid"),
    ("planner", "PLANNER", "Dijkstra cost grid"),
    ("copilot", "COPILOT", "Nemotron @ Nebius TF"),
]


def stack_panel(ms: dict, fps: float, device: str) -> "np.ndarray":
    img = blank()
    label(img, "Software stack // live")
    pw = img.shape[1]
    total = max(sum(v for k, v in ms.items() if k != "copilot"), 1e-3)   # copilot runs off-frame
    for i, (key, name, model) in enumerate(STAGES):
        y = 38 + i * 21
        t = ms.get(key, 0.0)
        text(img, name, 8, y, NEON, 0.34)
        text(img, model, 8, y + 11, GREY, 0.28)
        mono(img, f"{t:5.0f}ms", pw - 62, y + 4, ICE, 0.85)
        bx0, bx1 = 150, pw - 70
        cv2.rectangle(img, (bx0, y - 6), (bx1, y + 2), DIM, -1)
        cv2.rectangle(img, (bx0, y - 6), (bx0 + int((bx1 - bx0) * min(t / total, 1)), y + 2), NEON, -1)

    gpu = torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", "") if device == "cuda" else "CPU"
    cv2.line(img, (8, 212), (pw - 8, 212), DIM, 1)
    text(img, gpu.upper(), 8, 229, WHITE, 0.32)
    mono(img, f"{fps:4.1f} FPS", pw - 78, 230, ICE, 0.95)
    return img
