from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    CLIPModel = None
    CLIPProcessor = None


class StudentFeatureExtractor:
    """
    Phase-1 extractor with robust fallback:
      - CLIP embedding when transformers are available
      - Always append geometric + color statistics
    """

    def __init__(self, clip_model_name: str = "openai/clip-vit-base-patch32", device: str = "cuda:0"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.clip_enabled = CLIPModel is not None and CLIPProcessor is not None
        self.clip_dim = 512
        self.clip_model: Optional[CLIPModel] = None
        self.clip_processor: Optional[CLIPProcessor] = None
        if self.clip_enabled:
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            self.clip_model.eval()

    def output_dim(self) -> int:
        # 512 clip + 8 bbox geom + 24 color stats
        return self.clip_dim + 8 + 24

    def extract(self, frame_bgr: np.ndarray, bbox_xyxy: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
        h, w = frame_bgr.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((self.output_dim(),), dtype=np.float32)

        crop = frame_bgr[y1:y2, x1:x2]
        clip = self._clip(crop)
        geom = self._geom(x1, y1, x2, y2, w, h)
        col = self._color_stats(crop)
        return np.concatenate([clip, geom, col], axis=0).astype(np.float32)

    def _clip(self, crop_bgr: np.ndarray) -> np.ndarray:
        if not self.clip_enabled or self.clip_model is None or self.clip_processor is None:
            return np.zeros((self.clip_dim,), dtype=np.float32)
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inp = self.clip_processor(images=Image.fromarray(rgb), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            f = self.clip_model.get_image_features(**inp)
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)
        return f[0].detach().float().cpu().numpy()

    def _geom(self, x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> np.ndarray:
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        cx = (x1 + x2) * 0.5 / max(1.0, float(w))
        cy = (y1 + y2) * 0.5 / max(1.0, float(h))
        area = (bw * bh) / max(1.0, float(w * h))
        ar = bw / bh
        left = x1 / max(1.0, float(w))
        right = x2 / max(1.0, float(w))
        top = y1 / max(1.0, float(h))
        bottom = y2 / max(1.0, float(h))
        return np.array([cx, cy, area, ar, left, right, top, bottom], dtype=np.float32)

    def _color_stats(self, crop_bgr: np.ndarray) -> np.ndarray:
        chans = cv2.split(crop_bgr)
        feats = []
        for ch in chans:
            chf = ch.astype(np.float32)
            feats.extend(
                [
                    float(chf.mean()),
                    float(chf.std()),
                    float(np.percentile(chf, 25)),
                    float(np.percentile(chf, 50)),
                    float(np.percentile(chf, 75)),
                    float(chf.min()),
                    float(chf.max()),
                    float((chf > 200).mean()),
                ]
            )
        return np.array(feats, dtype=np.float32)

