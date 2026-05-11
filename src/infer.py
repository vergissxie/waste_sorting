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
    FOLDS,
    IMG_SIZE,
    NUM_CLASSES,
    NUM_WORKERS,
    PREDICTION_DIR,
    RESULT_FILE,
)
from dataset import TestDataset, get_eval_transform, read_test_names
from model import create_model


def load_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
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
    dataset = TestDataset(image_names, transform=get_eval_transform(args.img_size))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    models = []
    for fold in range(FOLDS):
        path = CHECKPOINT_DIR / f"convnext_tiny_fold{fold}_best.pth"
        if path.exists():
            models.append(load_model(path, device))
        else:
            print(f"Skip missing checkpoint: {path}")
    if len(models) != FOLDS and not args.allow_partial:
        raise FileNotFoundError(
            f"Expected {FOLDS} fold checkpoints, found {len(models)}. "
            "Use --allow-partial only for smoke tests."
        )
    if not models:
        raise FileNotFoundError("No fold checkpoints found.")

    predictions: list[tuple[str, int]] = []
    for images, names in tqdm(loader, desc="infer"):
        images = images.to(device, non_blocking=True)
        probs = torch.zeros(images.size(0), NUM_CLASSES, device=device)
        for model in models:
            logits = model(images)
            fold_probs = F.softmax(logits, dim=1)
            if not args.no_tta:
                flipped_logits = model(torch.flip(images, dims=[3]))
                fold_probs = (
                    fold_probs + F.softmax(flipped_logits, dim=1)
                ) / 2.0
            probs += fold_probs
        probs /= len(models)
        labels = probs.argmax(dim=1).cpu().tolist()
        predictions.extend(zip(names, labels))

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for image_name, label in predictions:
            f.write(f"{image_name}\t{label}\n")
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
