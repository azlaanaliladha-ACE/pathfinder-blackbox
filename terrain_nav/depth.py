"""Metric monocular depth with Depth Anything V2 (outdoor or indoor checkpoint, see config).

Turns a normal camera into a pseudo depth camera: output is metres per pixel.
https://github.com/DepthAnything/Depth-Anything-V2

fast=True (CUDA only): half-precision model and preprocessing done on the GPU instead of the
Hugging Face image processor. About 3x faster (56 -> 19 ms on an RTX 4050), for live use.
"""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

import config

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class DepthEstimator:
    def __init__(self, device: str, fast: bool = False, short_side: int = 518):
        self.device = device
        self.fast = fast and device == "cuda"
        self.short_side = short_side
        self.proc = AutoImageProcessor.from_pretrained(config.DEPTH_MODEL)
        self.model = AutoModelForDepthEstimation.from_pretrained(config.DEPTH_MODEL).to(device).eval()
        if self.fast:
            self.model = self.model.half()
            self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
            self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def estimate(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Return an HxW depth map in metres (larger = farther)."""
        h, w = frame_rgb.shape[:2]
        if self.fast:
            depth = self._fast(frame_rgb)
        else:
            inputs = self.proc(images=Image.fromarray(frame_rgb), return_tensors="pt").to(self.device)
            depth = self.model(**inputs).predicted_depth
        depth = torch.nn.functional.interpolate(depth[None].float(), size=(h, w), mode="bilinear")[0, 0]
        return np.clip(depth.cpu().numpy(), 0.1, config.MAX_DEPTH_M)

    def _fast(self, frame_rgb):
        h, w = frame_rgb.shape[:2]
        k = self.short_side / min(h, w)                      # keep aspect, sides multiple of 14 (ViT patch)
        size = (max(14, round(h * k / 14) * 14), max(14, round(w * k / 14) * 14))
        x = torch.from_numpy(frame_rgb).to(self.device).permute(2, 0, 1)[None].float() / 255
        x = torch.nn.functional.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = ((x - self.mean) / self.std).half()
        return self.model(pixel_values=x).predicted_depth
