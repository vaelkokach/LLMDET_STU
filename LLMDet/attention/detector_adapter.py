from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from mmdet.apis.inference import inference_detector, init_detector


@dataclass
class DetectionResult:
    bbox_xyxy: List[float]
    score: float
    label: int


class FrozenLLMDetAdapter:
    """
    Frozen student detector wrapper around trained LLMDet checkpoint.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        text_prompt: str = "student",
        score_thr: float = 0.35,
        device: str = "cuda:0",
        max_det: int = 100,
    ):
        cfg_path = Path(config_path)
        ckpt_path = Path(checkpoint_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        self.text_prompt = text_prompt
        self.score_thr = score_thr
        self.max_det = max_det
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = init_detector(str(cfg_path), str(ckpt_path), device=self.device)

    def detect(self, frame_bgr: np.ndarray) -> List[DetectionResult]:
        out = inference_detector(
            self.model,
            frame_bgr,
            text_prompt=self.text_prompt,
            custom_entities=True,
        )
        pred = out.pred_instances
        if pred is None or len(pred) == 0:
            return []

        bboxes = pred.bboxes.detach().cpu().numpy()
        scores = pred.scores.detach().cpu().numpy()
        labels = pred.labels.detach().cpu().numpy()

        keep = scores >= self.score_thr
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]
        if scores.size == 0:
            return []

        order = np.argsort(-scores)[: self.max_det]
        results: List[DetectionResult] = []
        for i in order:
            results.append(
                DetectionResult(
                    bbox_xyxy=[float(x) for x in bboxes[i].tolist()],
                    score=float(scores[i]),
                    label=int(labels[i]),
                )
            )
        return results

