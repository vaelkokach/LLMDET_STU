import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from attention.features import StudentFeatureExtractor


CLASS_NAMES = ["attentive", "distracted", "sleeping", "engaged"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASS_NAMES)}

WEAK_LABEL_MAP = {
    "focused": "attentive",
    "thinking": "attentive",
    "typing": "engaged",
    "engaged": "engaged",
    "looking at screen": "engaged",
    "curious": "engaged",
    "calm": "attentive",
    "resting hand": "distracted",
    "distracted": "distracted",
    "sleeping": "sleeping",
    "tired": "sleeping",
    "yawning": "sleeping",
}


@dataclass
class RegionObs:
    filename: str
    frame_idx: int
    timestamp: str
    video_id: str
    bbox_xyxy: List[float]
    label_id: int


def _extract_video_id(filename: str) -> str:
    m = re.search(r"(video_\d+)", filename)
    if m:
        return m.group(1)
    return "video_unknown"


def _extract_frame_idx(filename: str) -> int:
    m = re.search(r"_f(\d+)_", filename)
    if m:
        return int(m.group(1))
    m = re.search(r"f(\d+)", filename)
    return int(m.group(1)) if m else 0


def _extract_timestamp(filename: str) -> str:
    ts = re.findall(r"(\d{14})", filename)
    return ts[0] if ts else "00000000000000"


def _region_to_label(region_phrase: str, tags: List[str], caption: str) -> int:
    text = " ".join([region_phrase, caption] + tags).lower()
    scores = defaultdict(int)
    for key, cls in WEAK_LABEL_MAP.items():
        if key in text:
            scores[cls] += 1
    if not scores:
        return CLASS_TO_ID["attentive"]
    cls = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0][0]
    return CLASS_TO_ID[cls]


def _iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def parse_jsonl(jsonl_path: Path) -> Dict[str, List[RegionObs]]:
    by_video: Dict[str, List[RegionObs]] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            fname = item["filename"]
            video_id = _extract_video_id(fname)
            frame_idx = _extract_frame_idx(fname)
            ts = _extract_timestamp(fname)
            grounding = item.get("grounding", {})
            caption = grounding.get("caption", "")
            tags = item.get("tags", [])
            for reg in grounding.get("regions", []):
                bbox = reg.get("bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue
                phrase = reg.get("phrase", "")
                label_id = _region_to_label(phrase, tags, caption)
                by_video[video_id].append(
                    RegionObs(
                        filename=fname,
                        frame_idx=frame_idx,
                        timestamp=ts,
                        video_id=video_id,
                        bbox_xyxy=[float(x) for x in bbox],
                        label_id=label_id,
                    )
                )
    for vid in by_video:
        by_video[vid].sort(key=lambda x: (x.frame_idx, x.timestamp, x.filename))
    return by_video


def assign_tracks(video_regions: List[RegionObs], iou_thr: float = 0.35) -> Dict[int, List[RegionObs]]:
    tracks: Dict[int, List[RegionObs]] = defaultdict(list)
    last_bbox: Dict[int, List[float]] = {}
    next_tid = 1

    grouped_by_frame: Dict[Tuple[int, str, str], List[RegionObs]] = defaultdict(list)
    for obs in video_regions:
        grouped_by_frame[(obs.frame_idx, obs.timestamp, obs.filename)].append(obs)

    for _, frame_obs in sorted(grouped_by_frame.items(), key=lambda x: x[0]):
        used = set()
        for tid in list(last_bbox.keys()):
            best_j = -1
            best_iou = 0.0
            for j, obs in enumerate(frame_obs):
                if j in used:
                    continue
                ov = _iou(last_bbox[tid], obs.bbox_xyxy)
                if ov > best_iou:
                    best_iou = ov
                    best_j = j
            if best_j >= 0 and best_iou >= iou_thr:
                tracks[tid].append(frame_obs[best_j])
                last_bbox[tid] = frame_obs[best_j].bbox_xyxy
                used.add(best_j)

        for j, obs in enumerate(frame_obs):
            if j in used:
                continue
            tid = next_tid
            next_tid += 1
            tracks[tid].append(obs)
            last_bbox[tid] = obs.bbox_xyxy

    return tracks


def build_sequences(
    jsonl_path: Path,
    image_root: Path,
    output_dir: Path,
    min_track_len: int = 8,
    max_track_len: int = 128,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    seq_dir = output_dir / "sequences"
    seq_dir.mkdir(exist_ok=True)

    extractor = StudentFeatureExtractor()
    by_video = parse_jsonl(jsonl_path)
    sample_idx = 0
    meta_rows = []

    for video_id, obs_list in by_video.items():
        tracks = assign_tracks(obs_list)
        for tid, track_obs in tracks.items():
            if len(track_obs) < min_track_len:
                continue
            track_obs = track_obs[:max_track_len]
            feats = []
            labels = []
            for obs in track_obs:
                img_path = image_root / obs.filename
                if not img_path.exists():
                    continue
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                feat = extractor.extract(frame, obs.bbox_xyxy)
                feats.append(feat)
                labels.append(obs.label_id)
            if len(feats) < min_track_len:
                continue

            x = np.stack(feats, axis=0).astype(np.float32)
            y = int(np.bincount(np.array(labels, dtype=np.int64)).argmax())
            out_name = f"sample_{sample_idx:06d}.npz"
            np.savez_compressed(seq_dir / out_name, x=x, y=np.array(y, dtype=np.int64))
            meta_rows.append({"file": out_name, "video_id": video_id, "track_id": tid, "length": int(x.shape[0]), "label": y})
            sample_idx += 1

    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump({"num_samples": sample_idx, "class_names": CLASS_NAMES, "samples": meta_rows}, f, indent=2)
    print(f"Built {sample_idx} temporal samples in {seq_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Build temporal training sequences from student JSONL annotations.")
    p.add_argument("--jsonl", type=str, required=True, help="Path to annotation JSONL file.")
    p.add_argument("--image-root", type=str, required=True, help="Path to directory containing images.")
    p.add_argument("--output-dir", type=str, required=True, help="Output directory for NPZ sequences.")
    p.add_argument("--min-track-len", type=int, default=8)
    p.add_argument("--max-track-len", type=int, default=128)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_sequences(
        jsonl_path=Path(args.jsonl),
        image_root=Path(args.image_root),
        output_dir=Path(args.output_dir),
        min_track_len=args.min_track_len,
        max_track_len=args.max_track_len,
    )

