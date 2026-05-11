from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from tensorboardX import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    AMP,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EPOCHS,
    FOLDS,
    IMG_SIZE,
    LOG_DIR,
    LR,
    NUM_CLASSES,
    NUM_WORKERS,
    SEED,
    USE_WEIGHTED_CE,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
)
from dataset import (
    GarbageDataset,
    audit_data,
    build_train_samples,
    get_eval_transform,
    get_train_transform,
)
from model import create_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_scheduler(optimizer: AdamW, epochs: int, warmup_epochs: int):
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=epochs)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs))
    return SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )


def make_class_weights(labels: list[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = np.sqrt(len(labels) / (NUM_CLASSES * counts))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / max(1, total)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="val", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    val_loss = total_loss / max(1, total)
    val_acc = correct / max(1, total)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return val_loss, val_acc, macro_f1


def train_fold(
    fold: int,
    samples: list[tuple[Path, int]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp and device.type == "cuda")

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    train_labels = [label for _, label in train_samples]

    train_dataset = GarbageDataset(train_samples, get_train_transform(args.img_size))
    val_dataset = GarbageDataset(val_samples, get_eval_transform(args.img_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(pretrained=True).to(device)
    if args.weighted_ce:
        class_weights = make_class_weights(train_labels).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    writer = SummaryWriter(str(LOG_DIR / f"convnext_tiny_fold{fold}"))

    best_acc = -math.inf
    best_path = CHECKPOINT_DIR / f"convnext_tiny_fold{fold}_best.pth"
    last_path = CHECKPOINT_DIR / f"convnext_tiny_fold{fold}_last.pth"
    print(f"Fold {fold}: train={len(train_dataset)}, val={len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            args.max_train_batches,
        )
        val_loss, val_acc, macro_f1 = validate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            args.max_val_batches,
        )
        scheduler.step()

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/val_acc", val_acc, epoch)
        writer.add_scalar("metric/macro_f1", macro_f1, epoch)
        print(
            f"Fold {fold} Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} macro_f1={macro_f1:.4f}"
        )

        state = {
            "fold": fold,
            "epoch": epoch,
            "img_size": args.img_size,
            "state_dict": model.state_dict(),
            "val_acc": val_acc,
            "macro_f1": macro_f1,
        }
        torch.save(state, last_path)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(state, best_path)
            print(f"Saved best checkpoint: {best_path}")

    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Train one fold only.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--weighted-ce", action=argparse.BooleanOptionalAction, default=USE_WEIGHTED_CE)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=AMP)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(SEED)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    audit = audit_data()
    print("Data audit:", audit)
    if args.audit_only:
        return

    samples = build_train_samples()
    labels = [label for _, label in samples]
    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    folds = list(splitter.split(np.zeros(len(labels)), labels))

    selected_folds = range(FOLDS) if args.fold is None else [args.fold]
    for fold in selected_folds:
        train_idx, val_idx = folds[fold]
        train_fold(fold, samples, train_idx, val_idx, args)


if __name__ == "__main__":
    main()
