"""Check each module on one real frame and print timings.

    python smoke_test.py videos/trail.mp4
"""
import sys
import time

import cv2
import torch

import config
from terrain_nav.boxes3d import lift
from terrain_nav.depth import DepthEstimator
from terrain_nav.obstacles import ObstacleDetector
from terrain_nav.planner import build_cost_grid, plan
from terrain_nav.terrain import TerrainClassifier


def timed(name, fn, *args):
    fn(*args)                                   # warm-up
    t = time.time()
    out = fn(*args)
    print(f"[ok] {name:<10} {(time.time() - t) * 1000:6.1f} ms")
    return out


def main(path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, frame = cap.read()
    assert ok, f"could not read {path}"
    frame = cv2.resize(frame, config.FRAME_SIZE)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    terrain = TerrainClassifier(device)
    idx, conf, cost = timed("terrain", terrain.classify, rgb)
    print("     labels:\n", [[terrain.labels[i] for i in row] for row in idx])

    depth = timed("depth", DepthEstimator(device).estimate, rgb)
    print(f"     depth range {depth.min():.1f}-{depth.max():.1f} m")

    obstacles = timed("obstacles", ObstacleDetector(device).detect, frame)
    print("    ", [(o.label, round(o.conf, 2)) for o in obstacles])

    boxes = [lift(o, depth) for o in obstacles]
    print("[ok] boxes3d   ", [(b.obstacle.label, round(b.distance_m, 1)) for b in boxes])

    path = timed("planner", lambda: plan(build_cost_grid(cost, obstacles, frame.shape)))
    print("     path:", path)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "videos/trail.mp4")
