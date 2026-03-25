import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from attention.temporal_model import AttentionTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SequenceNPZDataset(Dataset):
    def __init__(self, seq_dir: Path):
        self.files = sorted(seq_dir.glob("*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz files found in {seq_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        item = np.load(self.files[idx])
        x = item["x"].astype(np.float32)
        y = int(item["y"].item()) if "y" in item else -1
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)

    def label_histogram(self, num_classes: int) -> np.ndarray:
        hist = np.zeros((num_classes,), dtype=np.int64)
        for fp in self.files:
            item = np.load(fp)
            if "y" not in item:
                continue
            y = int(item["y"].item())
            if 0 <= y < num_classes:
                hist[y] += 1
        return hist


def collate_varlen(batch):
    xs, ys = zip(*batch)
    max_t = max(x.shape[0] for x in xs)
    d = xs[0].shape[1]
    out = torch.zeros((len(xs), max_t, d), dtype=torch.float32)
    for i, x in enumerate(xs):
        out[i, : x.shape[0]] = x
    return out, torch.stack(ys)


def init_ddp(launcher: str) -> Tuple[bool, int, int, int]:
    if launcher == "none":
        return False, 0, 1, 0
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return True, rank, world_size, local_rank


def cleanup_ddp() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description="DDP temporal training for student attention.")
    p.add_argument("--config", type=str, required=True, help="YAML config path.")
    p.add_argument("--launcher", type=str, default="none", choices=["none", "pytorch"])
    return p.parse_args()


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target.tolist(), pred.tolist()):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def per_class_metrics(cm: np.ndarray) -> Dict[str, List[float]]:
    num_classes = cm.shape[0]
    precision, recall, f1 = [], [], []
    for c in range(num_classes):
        tp = float(cm[c, c])
        fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        p = tp / (tp + fp + 1e-6)
        r = tp / (tp + fn + 1e-6)
        f = 2.0 * p * r / (p + r + 1e-6)
        precision.append(p)
        recall.append(r)
        f1.append(f)
    return {"precision": precision, "recall": recall, "f1": f1}


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, class_weights: torch.Tensor):
    model.train()
    mode = cfg["training"]["mode"]
    pseudo_thr = float(cfg["training"].get("pseudo_label_threshold", 0.7))
    losses, accs = [], []

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=cfg["training"].get("amp", True) and device.type == "cuda"):
            logits = model(x)
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            valid = y >= 0

            loss = torch.tensor(0.0, device=device)
            total = 0
            if valid.any():
                loss = loss + F.cross_entropy(logits[valid], y[valid], weight=class_weights)
                total += 1

            if mode == "semi_supervised":
                unlabeled = ~valid
                if unlabeled.any():
                    conf = probs.max(dim=-1).values
                    pseudo_mask = unlabeled & (conf >= pseudo_thr)
                    if pseudo_mask.any():
                        pseudo = pred.detach()
                        loss = loss + F.cross_entropy(logits[pseudo_mask], pseudo[pseudo_mask], weight=class_weights)
                        total += 1
            if total == 0:
                continue
            loss = loss / total

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()

        losses.append(float(loss.item()))
        if valid.any():
            accs.append(float((pred[valid] == y[valid]).float().mean().item()))
    return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0


@torch.no_grad()
def validate(model, loader, device, class_weights: torch.Tensor, num_classes: int):
    model.eval()
    losses, accs = [], []
    all_pred = []
    all_tgt = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        valid = y >= 0
        if not valid.any():
            continue
        logits = model(x)
        loss = F.cross_entropy(logits[valid], y[valid], weight=class_weights)
        pred = logits.argmax(dim=-1)
        all_pred.append(pred[valid].detach().cpu().numpy())
        all_tgt.append(y[valid].detach().cpu().numpy())
        losses.append(float(loss.item()))
        accs.append(float((pred[valid] == y[valid]).float().mean().item()))
    if all_pred:
        pred_np = np.concatenate(all_pred, axis=0)
        tgt_np = np.concatenate(all_tgt, axis=0)
        cm = compute_confusion_matrix(pred_np, tgt_np, num_classes=num_classes)
        cls = per_class_metrics(cm)
    else:
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        cls = {"precision": [0.0] * num_classes, "recall": [0.0] * num_classes, "f1": [0.0] * num_classes}
    return (
        float(np.mean(losses)) if losses else 0.0,
        float(np.mean(accs)) if accs else 0.0,
        cm,
        cls,
    )


def main():
    args = parse_args()
    cfg = load_yaml(Path(args.config))
    set_seed(int(cfg.get("seed", 42)))

    ddp, rank, _, local_rank = init_ddp(args.launcher)
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    seq_root = Path(cfg["data"]["sequence_dir"])
    train_set = SequenceNPZDataset(seq_root / "train")
    val_set = SequenceNPZDataset(seq_root / "val")
    num_classes = int(cfg["model"]["num_classes"])
    use_class_weights = bool(cfg["training"].get("use_class_weights", True))
    class_hist = train_set.label_histogram(num_classes)
    val_hist = val_set.label_histogram(num_classes)
    required_min_per_class = int(cfg["training"].get("required_min_per_class", 1))
    enforce_full_class_coverage = bool(cfg["training"].get("enforce_full_class_coverage", True))
    missing = [i for i, c in enumerate(class_hist.tolist()) if c < required_min_per_class]
    if enforce_full_class_coverage and missing:
        raise RuntimeError(
            f"Insufficient class coverage in training data. class_hist={class_hist.tolist()}, "
            f"required_min_per_class={required_min_per_class}, missing_class_indices={missing}. "
            "Add/curate data for missing classes before 4-class training."
        )
    if use_class_weights:
        # Inverse-frequency weighting to mitigate mode collapse to dominant classes.
        safe = np.maximum(class_hist.astype(np.float32), 1.0)
        inv = 1.0 / safe
        inv = inv / np.sum(inv) * num_classes
        class_weights_np = inv.astype(np.float32)
    else:
        class_weights_np = np.ones((num_classes,), dtype=np.float32)
    class_weights = torch.from_numpy(class_weights_np).to(device)
    if is_main:
        print(
            f"[info] train class_hist={class_hist.tolist()} val class_hist={val_hist.tolist()} "
            f"class_weights={class_weights_np.tolist()}"
        )

    train_sampler = DistributedSampler(train_set, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_set,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_varlen,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False,
        sampler=val_sampler,
        num_workers=int(cfg["training"].get("num_workers", 4)),
        pin_memory=True,
        collate_fn=collate_varlen,
    )

    model = AttentionTransformer(
        input_dim=int(cfg["model"]["input_dim"]),
        hidden_dim=int(cfg["model"]["hidden_dim"]),
        num_layers=int(cfg["model"]["num_layers"]),
        num_heads=int(cfg["model"]["num_heads"]),
        dropout=float(cfg["model"]["dropout"]),
        num_classes=int(cfg["model"]["num_classes"]),
        max_seq_len=int(cfg["model"]["max_seq_len"]),
    ).to(device)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"].get("weight_decay", 0.01)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["training"].get("amp", True) and device.type == "cuda")

    out_dir = Path(cfg["training"]["output_dir"])
    ckpt_dir = out_dir / "checkpoints"
    tb_dir = out_dir / "tb"
    if is_main:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(tb_dir)) if is_main else None

    best_val = -1.0
    epochs = int(cfg["training"]["epochs"])
    for epoch in range(epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, class_weights)
        va_loss, va_acc, va_cm, va_cls = validate(model, val_loader, device, class_weights, num_classes)

        if is_main:
            print(f"[epoch {epoch}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")
            if writer is not None:
                writer.add_scalar("train/loss", tr_loss, epoch)
                writer.add_scalar("train/acc", tr_acc, epoch)
                writer.add_scalar("val/loss", va_loss, epoch)
                writer.add_scalar("val/acc", va_acc, epoch)
                for c in range(num_classes):
                    writer.add_scalar(f"val/class_{c}_precision", float(va_cls["precision"][c]), epoch)
                    writer.add_scalar(f"val/class_{c}_recall", float(va_cls["recall"][c]), epoch)
                    writer.add_scalar(f"val/class_{c}_f1", float(va_cls["f1"][c]), epoch)

            state = {
                "epoch": epoch,
                "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": cfg,
            }
            torch.save(state, ckpt_dir / f"epoch_{epoch:03d}.pth")
            if va_acc > best_val:
                best_val = va_acc
                torch.save(state, ckpt_dir / "best.pth")
                np.save(ckpt_dir / "best_val_confusion_matrix.npy", va_cm)
                with (ckpt_dir / "best_val_class_metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(va_cls, f, indent=2)

    if is_main and writer is not None:
        writer.close()
        with (out_dir / "train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_val_acc": best_val,
                    "train_class_hist": class_hist.tolist(),
                    "val_class_hist": val_hist.tolist(),
                    "class_weights": class_weights_np.tolist(),
                },
                f,
                indent=2,
            )

    cleanup_ddp()


if __name__ == "__main__":
    main()

