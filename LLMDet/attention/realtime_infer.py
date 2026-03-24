import argparse
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml

from attention.detector_adapter import FrozenLLMDetAdapter
from attention.features import StudentFeatureExtractor
from attention.temporal_model import AttentionTransformer, logits_to_pred
from attention.tracking import IoUTracker


CLASS_NAMES = ["attentive", "distracted", "sleeping", "engaged"]
CLASS_SCORE = {"attentive": 1.0, "engaged": 0.9, "distracted": 0.3, "sleeping": 0.1}


def parse_args():
    p = argparse.ArgumentParser(description="Real-time student attention inference.")
    p.add_argument("--config", type=str, required=True, help="YAML config path.")
    p.add_argument("--source", type=str, default="0", help="Webcam index or video path.")
    p.add_argument("--show", dest="show", action="store_true", help="Display live OpenCV window.")
    p.add_argument("--no-show", dest="show", action="store_false", help="Run headless without GUI display.")
    p.set_defaults(show=True)
    p.add_argument("--out-video", type=str, default="", help="Optional output video path for annotated frames.")
    return p.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _blur_faces(frame: np.ndarray, tracks) -> np.ndarray:
    out = frame.copy()
    for t in tracks:
        x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
        h = max(0, y2 - y1)
        y2f = y1 + int(0.35 * h)
        if x2 <= x1 or y2f <= y1:
            continue
        roi = out[max(0, y1):max(0, y2f), max(0, x1):max(0, x2)]
        if roi.size:
            out[max(0, y1):max(0, y2f), max(0, x1):max(0, x2)] = cv2.GaussianBlur(roi, (31, 31), 0)
    return out


def main():
    args = parse_args()
    cfg = load_yaml(Path(args.config))

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")
    writer = None
    if args.out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0 or np.isnan(fps):
            fps = 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            w, h = 1280, 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.out_video, fourcc, fps, (w, h))
        print(f"[info] writing annotated video to: {args.out_video}")

    det = FrozenLLMDetAdapter(
        config_path=cfg["detector"]["config_path"],
        checkpoint_path=cfg["detector"]["checkpoint_path"],
        text_prompt=cfg["detector"].get("text_prompt", "student"),
        score_thr=float(cfg["detector"].get("score_thr", 0.35)),
        device=cfg["detector"].get("device", "cuda:0"),
        max_det=int(cfg["detector"].get("max_det", 100)),
    )
    tracker = IoUTracker(iou_match_thr=float(cfg["tracking"].get("iou_match_thr", 0.35)), max_age=int(cfg["tracking"].get("max_age", 30)))
    feat = StudentFeatureExtractor(
        clip_model_name=cfg["features"].get("clip_model_name", "openai/clip-vit-base-patch32"),
        device=cfg["features"].get("device", "cuda:0"),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = AttentionTransformer(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=int(cfg["model"]["num_classes"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
    ).to(device)
    ckpt = torch.load(cfg["temporal_checkpoint"], map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    win = int(cfg["inference"]["window_size"])
    feats: Dict[int, Deque[np.ndarray]] = defaultdict(lambda: deque(maxlen=win))
    latest_score: Dict[int, float] = {}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        dets = det.detect(frame)
        tracks = tracker.update(dets)
        preds: Dict[int, Tuple[str, float]] = {}

        active_ids = set()
        for t in tracks:
            active_ids.add(t.track_id)
            f = feat.extract(frame, t.bbox_xyxy)
            feats[t.track_id].append(f)
            if len(feats[t.track_id]) < int(cfg["inference"].get("min_frames_for_pred", 4)):
                continue

            x = np.stack(list(feats[t.track_id]), axis=0).astype(np.float32)
            x = torch.from_numpy(x).unsqueeze(0).to(device)
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                logits = model(x)
                pred, conf = logits_to_pred(logits)
            label = CLASS_NAMES[int(pred.item())]
            score = float(conf.item())
            preds[t.track_id] = (label, score)
            latest_score[t.track_id] = CLASS_SCORE.get(label, score)

        for tid in list(feats.keys()):
            if tid not in active_ids:
                feats.pop(tid, None)
                latest_score.pop(tid, None)

        classroom = float(np.mean(list(latest_score.values()))) if latest_score else 0.0
        vis = _blur_faces(frame, tracks) if bool(cfg["privacy"].get("anonymize_faces", True)) else frame.copy()

        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox_xyxy]
            label, conf = preds.get(t.track_id, ("pending", 0.0))
            col = (0, 255, 0) if label in ("attentive", "engaged") else (0, 165, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, f"id={t.track_id} {label}:{conf:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        cv2.putText(vis, f"classroom_attention={classroom:.2f}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        if writer is not None:
            writer.write(vis)
        if args.show:
            try:
                cv2.imshow("LLMDet Attention Realtime", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            except cv2.error as e:
                print(f"[warn] display is unavailable, switching to headless mode: {e}")
                args.show = False

    cap.release()
    if writer is not None:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

