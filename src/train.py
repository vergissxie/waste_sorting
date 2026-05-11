from __future__ import annotations

import argparse
import copy
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    AMP,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CUTMIX_ALPHA,
    EMA,
    EMA_DECAY,
    EPOCHS,
    FOLDS,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LOG_DIR,
    LR,
    MIX_PROB,
    MIXUP_ALPHA,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_WORKERS,
    SAVE_METRIC,
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
    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs + 1) / max(1, epochs - warmup_epochs)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def make_class_weights(labels: list[int]) -> torch.Tensor:
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = np.sqrt(len(labels) / (NUM_CLASSES * counts))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def smooth_one_hot(labels: torch.Tensor, num_classes: int, smoothing: float) -> torch.Tensor:
    confidence = 1.0 - smoothing
    off_value = smoothing / num_classes
    targets = torch.full(
        (labels.size(0), num_classes),
        off_value,
        device=labels.device,
        dtype=torch.float32,
    )
    targets.scatter_(1, labels.unsqueeze(1), confidence + off_value)
    return targets


def soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=1)
    if class_weights is not None:
        loss = -(targets * log_probs * class_weights.unsqueeze(0)).sum(dim=1)
    else:
        loss = -(targets * log_probs).sum(dim=1)
    return loss.mean()


def rand_bbox(size: torch.Size, lam: float) -> tuple[int, int, int, int]:
    height = size[2]
    width = size[3]
    cut_ratio = math.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)
    cx = np.random.randint(width)
    cy = np.random.randint(height)
    x1 = np.clip(cx - cut_w // 2, 0, width)
    y1 = np.clip(cy - cut_h // 2, 0, height)
    x2 = np.clip(cx + cut_w // 2, 0, width)
    y2 = np.clip(cy + cut_h // 2, 0, height)
    return int(x1), int(y1), int(x2), int(y2)


def apply_mixup_cutmix(
    images: torch.Tensor,
    targets: torch.Tensor,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mix_prob <= 0 or random.random() > mix_prob:
        return images, targets

    batch_size = images.size(0)
    indices = torch.randperm(batch_size, device=images.device)
    use_cutmix = cutmix_alpha > 0 and (mixup_alpha <= 0 or random.random() < 0.5)

    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[indices, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / (images.size(2) * images.size(3)))
        return mixed, targets * lam + targets[indices] * (1.0 - lam)

    if mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        mixed = images * lam + images[indices] * (1.0 - lam)
        return mixed, targets * lam + targets[indices] * (1.0 - lam)

    return images, targets


class ModelEma:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        for param in self.module.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.module.state_dict()
        model_state = model.state_dict()
        for key, value in ema_state.items():
            model_value = model_state[key].detach()
            if value.dtype.is_floating_point:
                value.mul_(self.decay).add_(model_value, alpha=1.0 - self.decay)
            else:
                value.copy_(model_value)


def metric_value(val_acc: float, macro_f1: float, save_metric: str) -> float:
    if save_metric == "acc":
        return val_acc
    if save_metric == "macro_f1":
        return macro_f1
    if save_metric == "blend":
        return 0.5 * val_acc + 0.5 * macro_f1
    raise ValueError(f"Unsupported save metric: {save_metric}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    class_weights: torch.Tensor | None,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    label_smoothing: float,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
    ema: ModelEma | None = None,
    max_batches: int | None = None,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total = 0
    optimizer_steps = 0
    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = smooth_one_hot(labels, NUM_CLASSES, label_smoothing)
        images, targets = apply_mixup_cutmix(
            images, targets, mixup_alpha, cutmix_alpha, mix_prob
        )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = soft_cross_entropy(logits, targets, class_weights)
        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        did_step = scaler.get_scale() >= old_scale
        if did_step:
            optimizer_steps += 1
        if ema is not None and did_step:
            ema.update(model)

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
    return total_loss / max(1, total), optimizer_steps


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
    val_dataset = GarbageDataset(
        val_samples, get_eval_transform(args.img_size, args.eval_resize_ratio)
    )
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

    model = create_model(args.model_name, pretrained=True).to(device)
    if args.weighted_ce:
        train_class_weights = make_class_weights(train_labels).to(device)
    else:
        train_class_weights = None
    val_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema = ModelEma(model, args.ema_decay) if args.ema else None
    checkpoint_prefix = args.checkpoint_prefix or args.model_name
    writer = SummaryWriter(str(LOG_DIR / f"{checkpoint_prefix}_fold{fold}"))

    best_score = -math.inf
    best_path = CHECKPOINT_DIR / f"{checkpoint_prefix}_fold{fold}_best.pth"
    last_path = CHECKPOINT_DIR / f"{checkpoint_prefix}_fold{fold}_last.pth"
    print(f"Fold {fold}: train={len(train_dataset)}, val={len(val_dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, optimizer_steps = train_one_epoch(
            model,
            train_loader,
            train_class_weights,
            optimizer,
            scaler,
            device,
            use_amp,
            args.label_smoothing,
            args.mixup_alpha,
            args.cutmix_alpha,
            args.mix_prob,
            ema,
            args.max_train_batches,
        )
        eval_model = ema.module if ema is not None else model
        val_loss, val_acc, macro_f1 = validate(
            eval_model,
            val_loader,
            val_criterion,
            device,
            use_amp,
            args.max_val_batches,
        )
        if optimizer_steps > 0:
            scheduler.step()
        score = metric_value(val_acc, macro_f1, args.save_metric)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("metric/val_acc", val_acc, epoch)
        writer.add_scalar("metric/macro_f1", macro_f1, epoch)
        writer.add_scalar(f"metric/{args.save_metric}", score, epoch)
        print(
            f"Fold {fold} Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} macro_f1={macro_f1:.4f} "
            f"{args.save_metric}={score:.4f}"
        )

        state = {
            "model_name": args.model_name,
            "checkpoint_prefix": checkpoint_prefix,
            "fold": fold,
            "epoch": epoch,
            "img_size": args.img_size,
            "eval_resize_ratio": args.eval_resize_ratio,
            "state_dict": eval_model.state_dict(),
            "val_acc": val_acc,
            "macro_f1": macro_f1,
            "score": score,
            "save_metric": args.save_metric,
            "ema": ema is not None,
        }
        torch.save(state, last_path)
        if score > best_score:
            best_score = score
            torch.save(state, best_path)
            print(f"Saved best checkpoint: {best_path}")

    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--checkpoint-prefix", type=str, default=None)
    parser.add_argument("--fold", type=int, default=None, help="Train one fold only.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--eval-resize-ratio", type=float, default=1.14)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--weighted-ce", action=argparse.BooleanOptionalAction, default=USE_WEIGHTED_CE)
    parser.add_argument("--label-smoothing", type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--mixup-alpha", type=float, default=MIXUP_ALPHA)
    parser.add_argument("--cutmix-alpha", type=float, default=CUTMIX_ALPHA)
    parser.add_argument("--mix-prob", type=float, default=MIX_PROB)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=EMA)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    parser.add_argument(
        "--save-metric",
        choices=["acc", "macro_f1", "blend"],
        default=SAVE_METRIC,
    )
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
