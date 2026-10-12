"""Terrain classification: CLIP zero-shot on each cell of a grid laid over the frame."""
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import config


def _unit(features) -> torch.Tensor:
    """L2-normalise CLIP features (transformers 5 returns an output object, 4.x a tensor)."""
    if not isinstance(features, torch.Tensor):
        features = features.pooler_output
    return features / features.norm(dim=-1, keepdim=True)


class TerrainClassifier:
    def __init__(self, device: str):
        self.device = device
        self.labels = list(config.TERRAIN)
        self.costs = np.array([config.TERRAIN[l][1] for l in self.labels])
        self.model = CLIPModel.from_pretrained(config.CLIP_MODEL).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
        prompts = [config.TERRAIN[l][0] for l in self.labels]
        with torch.no_grad():
            tok = self.proc(text=prompts, return_tensors="pt", padding=True).to(device)
            self.text = _unit(self.model.get_text_features(**tok))

    @torch.no_grad()
    def classify(self, frame_rgb: np.ndarray):
        """Return (label_idx grid, confidence grid, cost grid), each GRID_ROWS x GRID_COLS."""
        h, w = frame_rgb.shape[:2]
        ch, cw = h // config.GRID_ROWS, w // config.GRID_COLS
        crops = [Image.fromarray(frame_rgb[r*ch:(r+1)*ch, c*cw:(c+1)*cw])
                 for r in range(config.GRID_ROWS) for c in range(config.GRID_COLS)]
        pix = self.proc(images=crops, return_tensors="pt")["pixel_values"].to(self.device)
        img = _unit(self.model.get_image_features(pixel_values=pix))
        shape = (config.GRID_ROWS, config.GRID_COLS, len(self.labels))
        probs = (100.0 * img @ self.text.T).softmax(dim=-1).cpu().numpy().reshape(shape)
        probs = self._apply_priors(probs)

        idx = probs.argmax(axis=-1)
        conf = probs.max(axis=-1)
        return idx, conf, self.costs[idx]

    def _apply_priors(self, probs: np.ndarray) -> np.ndarray:
        """Remove physically implausible or low-confidence impassable labels."""
        probs = probs.copy()
        sky = self.labels.index("sky")
        probs[int(config.GRID_ROWS * config.SKY_MAX_ROW_FRAC):, :, sky] = 0   # no sky on the ground
        impassable = np.isinf(self.costs)
        weak = probs[..., impassable] < config.IMPASSABLE_MIN_CONF
        probs[..., impassable] = np.where(weak, 0, probs[..., impassable])
        return probs / probs.sum(axis=-1, keepdims=True)
