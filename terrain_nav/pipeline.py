"""One frame in -> one annotated frame out. Shared by the desktop app, recorder and website.

Layout (1280x720), black / neon-blue theme:
    +----------------------+----------------------+
    |  camera feed:        |  depth map           |
    |  terrain, boxes,     |  (Depth Anything V2) |
    |  path                |                      |
    +----------+-----------+----------+-----------+
    | 1 world  | 2 terrain | 3 Nemo-  | 4 software|
    |   map    |   stats   |   tron   |   stack   |
    +----------+-----------+----------+-----------+

Panel 3 is the Nemotron copilot (Nebius Token Factory) when NEBIUS_API_KEY is set,
otherwise the simulated drone telemetry.
"""
import time

import cv2
import numpy as np
import torch

import config
from terrain_nav.boxes3d import lift
from terrain_nav.copilot import Copilot, scene_summary
from terrain_nav.copilot_view import copilot_panel, draw_callout
from terrain_nav.costmap import costmap_panel
from terrain_nav.depth import DepthEstimator
from terrain_nav.depthview import DepthView
from terrain_nav.drone import DroneTelemetry, telemetry_panel
from terrain_nav.geometry import PlaneFilter, backproject, drivable_points
from terrain_nav.ground_track import GroundTracker
from terrain_nav.hud import draw_box3d, draw_path, draw_status, draw_terrain
from terrain_nav.obstacles import ObstacleDetector
from terrain_nav.panels import ICE, finish
from terrain_nav.planner import build_cost_grid, plan
from terrain_nav.stack import stack_panel
from terrain_nav.terrain import TerrainClassifier
from terrain_nav.world_map import GroundOdometry, WorldMap

MAP_STRIDE = 4                      # sample every Nth pixel of the whole frame for the world map


class Pipeline:
    def __init__(self, live: bool = True, mission: dict | None = None):
        self.live = live                # live: copilot calls on a thread; recording: inline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading models on {self.device} ...")
        self.terrain = TerrainClassifier(self.device)
        self.depth = DepthEstimator(self.device)
        self.detector = ObstacleDetector(self.device)
        self.drone = DroneTelemetry()
        self.tracker = GroundTracker(self.device)
        self.odometry = GroundOdometry()
        self.world = WorldMap(self.terrain.labels)
        self.planes = PlaneFilter()
        self.depth_view = DepthView()
        self.sky = self.terrain.labels.index("sky")
        self.copilot = Copilot(mission)
        print(f"Copilot ({self.copilot.model}): {self.copilot.status}")
        self.reset()

    def reset(self, source_fps: float = 30.0):
        """Call when switching to a new video so cached state is not reused."""
        self.frame_no, self.terrain_out, self.fps, self.last_path = 0, None, 0.0, None
        self.dt = 1.0 / source_fps
        self.t_start = time.time()
        self.ms = {}
        self.depth_view = DepthView()
        self.drone.reset()
        self.tracker.reset()
        self.odometry.reset()
        self.planes.reset()
        self.world.reset()
        self.copilot.reset()

    @property
    def flight_time(self) -> float:
        """Seconds into the run: wall clock when live, video time when recording."""
        return time.time() - self.t_start if self.live else self.frame_no * self.dt

    def _timed(self, key, fn, *args):
        t = time.perf_counter()
        out = fn(*args)
        self.ms[key] = 0.8 * self.ms.get(key, 0.0) + 0.2 * (time.perf_counter() - t) * 1000
        return out

    def process(self, frame):
        t0 = time.time()
        frame = cv2.resize(frame, config.FRAME_SIZE)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- perception ---
        if self.terrain_out is None or self.frame_no % config.TERRAIN_EVERY_N_FRAMES == 0:
            self.terrain_out = self._timed("terrain", self.terrain.classify, rgb)
        idx, conf, terrain_cost = self.terrain_out
        depth = self._timed("depth", self.depth.estimate, rgb)
        obstacles = self._timed("objects", self.detector.detect, frame)
        boxes = [lift(o, depth) for o in obstacles]
        path = self._timed("planner", lambda: plan(build_cost_grid(terrain_cost, obstacles, frame.shape)))
        self.last_path = path

        # --- geometry + state ---
        u, v, pts = drivable_points(depth, config.CLOUD_STRIDE)
        plane = self.planes.update(pts)
        track = self._timed("tracking", self.tracker.update, rgb)
        shift = track.shift if track is not None else None
        self.drone.update(frame, plane, self.dt, track_shift=shift)
        T = self._timed("odometry", self.odometry.update, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), depth, plane)
        self._timed("map", self._update_world, depth, plane, idx, T)

        # --- reasoning: Nemotron on Nebius Token Factory reads the scene summary ---
        scene = scene_summary(self.terrain.labels, idx, terrain_cost, boxes, path, self.drone.s)
        self.copilot.maybe_decide(scene, self.flight_time, blocking=not self.live)
        if self.copilot.last.latency_ms:
            self.ms["copilot"] = self.copilot.last.latency_ms

        self.fps = 0.9 * self.fps + 0.1 / max(time.time() - t0, 1e-3)
        panels = np.hstack([finish(p) for p in (
            self.world.render(),
            costmap_panel(idx, self.terrain.labels, terrain_cost, path),
            copilot_panel(self.copilot, self.drone) if self.copilot.enabled else telemetry_panel(self.drone),
            stack_panel(self.ms, self.fps, self.device),
        )])

        # --- top row: camera feed | depth map ---
        draw_terrain(frame, self.terrain.labels, idx, terrain_cost, config.PLAN_FIRST_ROW)
        if track is not None:
            for x, y in track.points.astype(int):
                cv2.circle(frame, (x, y), 2, ICE, -1, cv2.LINE_AA)
        for b in boxes:
            draw_box3d(frame, b)
        draw_path(frame, path, idx.shape)
        draw_callout(frame, self.copilot)
        draw_status(frame, "CAMERA FEED  //  TERRAIN + PATH",
                    f"{self.fps:4.1f} FPS   OBSTACLES {len(obstacles):02d}")
        near = depth[int(depth.shape[0] * 0.6):].min()
        dview = self.depth_view.render(depth, f"NEAREST {near:4.1f} M")

        self.frame_no += 1
        return np.vstack([np.hstack([frame, dview]), panels])

    def _update_world(self, depth, plane, idx, T):
        """Add everything the camera sees (ground, cliffs, ridges; not sky) to the world map."""
        h, w = depth.shape
        v, u = np.mgrid[0:h:MAP_STRIDE, 0:w:MAP_STRIDE]
        u, v = u.ravel(), v.ravel()
        rows, cols = idx.shape
        cell = idx[np.minimum(v * rows // h, rows - 1), np.minimum(u * cols // w, cols - 1)]
        z = depth[v, u]
        keep = (z < config.MAP_MAX_DEPTH_M) & (cell != self.sky)
        pts = backproject(u[keep], v[keep], z[keep], w, h)
        elev = np.maximum(plane.elevation(pts), config.MAP_MIN_ELEV_M)
        self.world.add_frame(plane.level(pts), cell[keep], elev, T, plane.camera_height)
