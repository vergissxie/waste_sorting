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
    SUPPORTED_MODELS,
)
from dataset import (
    TestDataset,
    get_eval_center_crop_transform,
    get_eval_transform,
    read_test_names,
)
from model import create_model


def parse_tta_scales(value: str) -> list[int]:
    scales = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not scales:
        raise argparse.ArgumentTypeError("At least one TTA scale is required.")
    if any(scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("TTA scales must be positive integers.")
    return scales


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    model_name: str,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def build_eval_transform(args: argparse.Namespace, img_size: int):
    if args.center_crop_resize is not None:
        return get_eval_center_crop_transform(
            img_size=img_size,
            resize_size=args.center_crop_resize,
        )
    return get_eval_transform(
        img_size=img_size,
        resize_ratio=args.eval_resize_ratio,
        mode=args.eval_transform_mode,
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument(
        "--tta-scales",
        "--tta-sizes",
        dest="tta_scales",
        type=parse_tta_scales,
        default=None,
        help="Comma-separated eval image sizes. Defaults to --img-size only.",
    )
    parser.add_argument("--eval-resize-ratio", type=float, default=EVAL_RESIZE_RATIO)
    parser.add_argument(
        "--eval-transform-mode",
        choices=["resize", "crop"],
        default="crop",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output", type=Path, default=RESULT_FILE)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument(
        "--model-name",
        type=str,
        choices=SUPPORTED_MODELS,
        default=MODEL_NAME,
    )
    parser.add_argument("--checkpoint-prefix", type=str, default=None)
    parser.add_argument("--checkpoint-suffix", type=str, default="best")
    parser.add_argument(
        "--center-crop-resize",
        type=int,
        default=None,
        help="Resize shorter side before CenterCrop(--img-size). Disabled by default.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow inference with fewer than 5 fold checkpoints for smoke tests.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_names = read_test_names()
    tta_scales = args.tta_scales or [args.img_size]
    checkpoint_prefix = args.checkpoint_prefix or args.model_name

    models = []
    for fold in range(FOLDS):
        checkpoint_path = CHECKPOINT_DIR / (
            f"{checkpoint_prefix}_fold{fold}_{args.checkpoint_suffix}.pth"
        )
        if checkpoint_path.exists():
            models.append(load_model(checkpoint_path, device, args.model_name))
        else:
            print(f"Skip missing checkpoint: {checkpoint_path}")

    if len(models) != FOLDS and not args.allow_partial:
        raise FileNotFoundError(
            f"Expected {FOLDS} fold checkpoints, found {len(models)}. "
            "Use --allow-partial only for smoke tests."
        )
    if not models:
        raise FileNotFoundError("No fold checkpoints found.")

    all_probs = torch.zeros(len(image_names), NUM_CLASSES, device=device)
    for img_size in tta_scales:
        dataset = TestDataset(
            image_names,
            transform=build_eval_transform(args, img_size),
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )

        offset = 0
        for images, _names in tqdm(loader, desc=f"infer {img_size}"):
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
            batch_size = images.size(0)
            all_probs[offset : offset + batch_size] += probs
            offset += batch_size

    all_probs /= len(tta_scales)
    labels = all_probs.argmax(dim=1).cpu().tolist()

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as f:
        for image_name, label in zip(image_names, labels):
            f.write(f"{image_name}\t{label}\n")
    print(f"Wrote {len(labels)} predictions to {args.output}")


if __name__ == "__main__":
    main()
