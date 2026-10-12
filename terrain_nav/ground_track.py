"""Ground-point tracking with TAPNet (BootsTAPIR, PyTorch) to measure camera motion.

A grid of points on the drivable ground is tracked through a sliding window of frames.
Ground points move with the camera, so their median image motion gives the camera's
real per-frame shift. The vehicle is never queried or tracked.

https://github.com/google-deepmind/tapnet
"""
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch

import config
from tapnet.torch import tapir_model

RES = 256            # TAPIR works on a 256x256 input
WINDOW = 8           # frames per tracking window
GRID_X, GRID_Y = 12, 8


@dataclass
class TrackState:
    points: np.ndarray            # [N, 2] current positions (full-res pixels), visible only
    shift: tuple | None           # (dx, dy) camera-induced pixel shift per frame, full-res


class GroundTracker:
    def __init__(self, device: str):
        self.device = device
        self.model = tapir_model.TAPIR(pyramid_level=1)
        self.model.load_state_dict(torch.load(config.TAPNET_CHECKPOINT, map_location="cpu"))
        self.model = self.model.to(device).eval()
        self.buf = deque(maxlen=WINDOW)
        self.query = self._grid_query()

    def _grid_query(self) -> torch.Tensor:
        """Points on the lower (drivable) part of the frame, as (t, y, x) at 256 resolution."""
        ys = np.linspace(RES * 0.45, RES - 8, GRID_Y)
        xs = np.linspace(8, RES - 8, GRID_X)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        q = np.stack([np.zeros(yy.size), yy.ravel(), xx.ravel()], axis=1).astype(np.float32)
        return torch.tensor(q, device=self.device)

    def reset(self):
        self.buf.clear()

    @torch.no_grad()
    def update(self, frame_rgb: np.ndarray) -> TrackState | None:
        self.buf.append(cv2.resize(frame_rgb, (RES, RES), interpolation=cv2.INTER_AREA))
        if len(self.buf) < WINDOW:
            return None

        frames = torch.tensor(np.stack(self.buf), device=self.device).float() / 255 * 2 - 1
        out = self.model(frames[None], self.query[None])
        tracks = out["tracks"][0]                                   # [N, T, 2] in (x, y)
        visible = (1 - torch.sigmoid(out["occlusion"][0])) * (1 - torch.sigmoid(out["expected_dist"][0])) > 0.5

        tracks = tracks.cpu().numpy()
        visible = visible.cpu().numpy()
        ok = visible[:, 0] & visible[:, -1]
        if ok.sum() < 8:
            return TrackState(np.empty((0, 2)), None)

        scale = config.FRAME_SIZE[0] / RES
        motion = np.median(tracks[ok, -1] - tracks[ok, 0], axis=0) / (WINDOW - 1)   # px/frame at 256
        shift = (float(motion[0] * scale), float(motion[1] * scale))
        points = tracks[ok, -1] * scale
        return TrackState(points, shift)
