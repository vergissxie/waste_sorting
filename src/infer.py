from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    EVAL_RESIZE_RATIO,
    FOLDS,
    IMG_SIZE,
    MODEL_NAME,
    NUM_CLASSES,
    NUM_WORKERS,
    PREDICTION_DIR,
    RESULT_FILE,
)
from dataset import TestDataset, get_eval_transform, read_test_names
from model import create_model


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_name = checkpoint.get("model_name", MODEL_NAME)
    model = create_model(model_name, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def parse_tta_sizes(value: str) -> list[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes:
        raise ValueError("At least one TTA size is required.")
    return sizes


def parse_csv_strings(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("At least one value is required.")
    return items


def parse_weights(value: str | None, count: int) -> list[float]:
    if value is None:
        return [1.0 / count] * count
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != count:
        raise ValueError("The number of ensemble weights must match prefixes.")
    total = sum(weights)
    if total <= 0:
        raise ValueError("Ensemble weights must sum to a positive value.")
    return [weight / total for weight in weights]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--checkpoint-prefix", type=str, default=None)
    parser.add_argument(
        "--checkpoint-prefixes",
        type=str,
        default=None,
        help="Comma-separated prefixes for model fusion.",
    )
    parser.add_argument(
        "--ensemble-weights",
        type=str,
        default=None,
        help="Comma-separated weights matching --checkpoint-prefixes.",
    )
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument(
        "--tta-sizes",
        type=str,
        default=None,
        help="Comma-separated inference sizes, for example 256,288,320.",
    )
    parser.add_argument("--eval-resize-ratio", type=float, default=EVAL_RESIZE_RATIO)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output", type=Path, default=RESULT_FILE)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow inference with fewer than 5 fold checkpoints for smoke tests.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_names = read_test_names()
    tta_sizes = parse_tta_sizes(args.tta_sizes) if args.tta_sizes else [args.img_size]

    prefixes = (
        parse_csv_strings(args.checkpoint_prefixes)
        if args.checkpoint_prefixes
        else [args.checkpoint_prefix or args.model_name]
    )
    prefix_weights = parse_weights(args.ensemble_weights, len(prefixes))
    models: list[tuple[torch.nn.Module, float]] = []
    for prefix, prefix_weight in zip(prefixes, prefix_weights):
        for fold in range(FOLDS):
            path = CHECKPOINT_DIR / f"{prefix}_fold{fold}_best.pth"
            if path.exists():
                models.append((load_model(path, device), prefix_weight / FOLDS))
            else:
                print(f"Skip missing checkpoint: {path}")
    expected_models = FOLDS * len(prefixes)
    if len(models) != expected_models and not args.allow_partial:
        raise FileNotFoundError(
            f"Expected {expected_models} checkpoints, found {len(models)}. "
            "Use --allow-partial only for smoke tests."
        )
    if not models:
        raise FileNotFoundError("No fold checkpoints found.")

    all_probs = torch.zeros(len(image_names), NUM_CLASSES, dtype=torch.float32)
    for size in tta_sizes:
        dataset = TestDataset(
            image_names,
            transform=get_eval_transform(size, args.eval_resize_ratio),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        offset = 0
        for images, names in tqdm(loader, desc=f"infer@{size}"):
            images = images.to(device, non_blocking=True)
            probs = torch.zeros(images.size(0), NUM_CLASSES, device=device)
            weight_sum = 0.0
            for model, weight in models:
                logits = model(images)
                fold_probs = F.softmax(logits, dim=1)
                if not args.no_tta:
                    flipped_logits = model(torch.flip(images, dims=[3]))
                    fold_probs = (
                        fold_probs + F.softmax(flipped_logits, dim=1)
                    ) / 2.0
                probs += fold_probs * weight
                weight_sum += weight
            probs /= weight_sum
            batch_size = images.size(0)
            all_probs[offset : offset + batch_size] += probs.cpu()
            offset += batch_size

    all_probs /= len(tta_sizes)
    labels = all_probs.argmax(dim=1).tolist()
    predictions = list(zip(image_names, labels))

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for image_name, label in predictions:
            f.write(f"{image_name}\t{label}\n")
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
