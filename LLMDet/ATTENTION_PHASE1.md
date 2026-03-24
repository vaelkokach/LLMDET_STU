# Attention Pipeline Phase 1 (Frozen LLMDet)

This phase keeps your detector frozen (`iter_15000.pth`) and trains only the temporal attention model.

## 1) Build temporal sequence dataset

Run from `LLMDet/`:

```bash
python attention_build_sequences.py \
  --jsonl ../grounding_data/stu_img/student_behavior_emotion_vg7_train.jsonl \
  --image-root ../grounding_data/stu_img \
  --output-dir ../grounding_data/stu_img/attention_sequences/train
```

```bash
python attention_build_sequences.py \
  --jsonl ../grounding_data/stu_img/student_behavior_emotion_vg7_val.jsonl \
  --image-root ../grounding_data/stu_img \
  --output-dir ../grounding_data/stu_img/attention_sequences/val
```

## 2) Train temporal model (single GPU)

```bash
python attention_train.py --config configs/attention_temporal.yaml --launcher none
```

## 3) Train temporal model (DDP / multi-GPU)

```bash
bash dist_attention_train.sh configs/attention_temporal.yaml 8
```

## 4) Real-time inference

```bash
python attention_realtime.py --config configs/attention_temporal.yaml --source 0
```

Use `--source /path/to/video.mp4` for video files.

## Notes

- Detector checkpoint is loaded from:
  `work_dirs/grounding_dino_swin_t/iter_15000.pth`
- Config is at:
  `configs/attention_temporal.yaml`
- Face anonymization for visualization is enabled by default.
- The sequence builder uses timestamp/frame info encoded in filenames and weak labels from phrase/tags.

