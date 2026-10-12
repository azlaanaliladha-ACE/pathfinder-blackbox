"""Obstacle detection with open-vocabulary YOLO-World (rocks, trees, logs, ...)."""
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

import config


@dataclass
class Obstacle:
    label: str
    conf: float
    box: tuple[int, int, int, int]      # x1, y1, x2, y2 in pixels


class ObstacleDetector:
    def __init__(self, device: str):
        self.device = device
        self.model = YOLO(config.YOLO_WEIGHTS)
        self.model.set_classes(config.OBSTACLE_CLASSES)

    def detect(self, frame_bgr: np.ndarray) -> list[Obstacle]:
        h, w = frame_bgr.shape[:2]
        res = self.model.predict(frame_bgr, conf=config.YOLO_CONF, device=self.device, verbose=False)[0]
        found = [Obstacle(res.names[int(b.cls)], float(b.conf), tuple(int(v) for v in b.xyxy[0]))
                 for b in res.boxes]
        return [o for o in found if not _is_ego_vehicle(o, w, h) and not _is_whole_scene(o, w, h)]


def _is_ego_vehicle(o: Obstacle, w: int, h: int) -> bool:
    x1, y1, x2, y2 = o.box
    return (y2 >= h * config.EGO_BOTTOM_FRAC and y1 >= h * config.EGO_MIN_TOP_FRAC
            and (x2 - x1) >= w * config.EGO_MIN_WIDTH_FRAC)


def _is_whole_scene(o: Obstacle, w: int, h: int) -> bool:
    x1, y1, x2, y2 = o.box
    return ((x2 - x1) * (y2 - y1) >= config.MAX_BOX_AREA_FRAC * w * h
            or (x2 - x1) >= config.MAX_BOX_WIDTH_FRAC * w)
