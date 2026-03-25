import argparse
import json
import shutil
from pathlib import Path


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def parse_args():
    p = argparse.ArgumentParser(description="Collect thesis artifacts into one reproducible bundle.")
    p.add_argument("--bundle-dir", type=str, default="work_dirs/thesis_bundle")
    p.add_argument("--e0-ckpt", type=str, default="work_dirs/grounding_dino_swin_t/iter_15000.pth")
    p.add_argument("--e1-best", type=str, required=True)
    p.add_argument("--e2-best", type=str, required=True)
    p.add_argument("--temporal-best", type=str, default="work_dirs/attention_temporal/checkpoints/best.pth")
    p.add_argument("--temporal-metrics-json", type=str, default="work_dirs/attention_temporal/checkpoints/best_val_class_metrics.json")
    p.add_argument("--temporal-cm-npy", type=str, default="work_dirs/attention_temporal/checkpoints/best_val_confusion_matrix.npy")
    p.add_argument("--ablation-root", type=str, default="work_dirs/ablation_e0_e1_e2")
    p.add_argument("--hybrid-video", type=str, default="work_dirs/attention_temporal/realtime_hybrid.mp4")
    return p.parse_args()


def main():
    args = parse_args()
    bundle = Path(args.bundle_dir)
    bundle.mkdir(parents=True, exist_ok=True)

    manifest = {
        "e0_ckpt": args.e0_ckpt,
        "e1_best": args.e1_best,
        "e2_best": args.e2_best,
        "temporal_best": args.temporal_best,
        "temporal_metrics_json": args.temporal_metrics_json,
        "temporal_cm_npy": args.temporal_cm_npy,
        "ablation_root": args.ablation_root,
        "hybrid_video": args.hybrid_video,
    }

    copy_if_exists(Path(args.e0_ckpt), bundle / "checkpoints" / "e0_iter15000.pth")
    copy_if_exists(Path(args.e1_best), bundle / "checkpoints" / "e1_best.pth")
    copy_if_exists(Path(args.e2_best), bundle / "checkpoints" / "e2_best.pth")
    copy_if_exists(Path(args.temporal_best), bundle / "checkpoints" / "temporal_best.pth")
    copy_if_exists(Path(args.temporal_metrics_json), bundle / "metrics" / "best_val_class_metrics.json")
    copy_if_exists(Path(args.temporal_cm_npy), bundle / "metrics" / "best_val_confusion_matrix.npy")
    copy_if_exists(Path(args.ablation_root), bundle / "ablation")
    copy_if_exists(Path(args.hybrid_video), bundle / "qualitative" / "realtime_hybrid.mp4")

    with (bundle / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Thesis bundle written to: {bundle}")


if __name__ == "__main__":
    main()
