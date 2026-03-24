from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from attention.detector_adapter import DetectionResult


@dataclass
class Track:
    track_id: int
    bbox_xyxy: List[float]
    score: float


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    inter = w * h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


class IoUTracker:
    """
    Lightweight online tracker for Phase 1.
    """

    def __init__(self, iou_match_thr: float = 0.35, max_age: int = 30):
        self.iou_match_thr = iou_match_thr
        self.max_age = max_age
        self.next_id = 1
        self.tracks: Dict[int, List[float]] = {}
        self.ages: Dict[int, int] = {}

    def update(self, detections: List[DetectionResult]) -> List[Track]:
        det_used = set()
        matched: List[Tuple[int, int]] = []
        active_ids = list(self.tracks.keys())

        for tid in active_ids:
            best_det = -1
            best_iou = 0.0
            for di, det in enumerate(detections):
                if di in det_used:
                    continue
                ov = iou_xyxy(self.tracks[tid], det.bbox_xyxy)
                if ov > best_iou:
                    best_iou = ov
                    best_det = di
            if best_det >= 0 and best_iou >= self.iou_match_thr:
                matched.append((tid, best_det))
                det_used.add(best_det)

        for tid, di in matched:
            self.tracks[tid] = detections[di].bbox_xyxy
            self.ages[tid] = 0

        for di, det in enumerate(detections):
            if di in det_used:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = det.bbox_xyxy
            self.ages[tid] = 0

        remove = []
        matched_ids = {tid for tid, _ in matched}
        for tid in self.tracks:
            if tid not in matched_ids:
                self.ages[tid] = self.ages.get(tid, 0) + 1
            if self.ages[tid] > self.max_age:
                remove.append(tid)
        for tid in remove:
            self.tracks.pop(tid, None)
            self.ages.pop(tid, None)

        output = []
        for tid, bbox in self.tracks.items():
            output.append(Track(track_id=tid, bbox_xyxy=bbox, score=1.0))
        return output

