from __future__ import annotations

import argparse
from dataclasses import dataclass
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
    SUPPORTED_MODELS,
)
from dataset import (
    TestDataset,
    get_eval_center_crop_transform,
    get_eval_transform,
    read_test_names,
)
from model import create_model


@dataclass(frozen=True)
class EnsembleMember:
    model_name: str
    checkpoint_prefix: str
    weight: float
    checkpoint_suffix: str


def parse_tta_scales(value: str) -> list[int]:
    scales = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not scales:
        raise argparse.ArgumentTypeError("At least one TTA scale is required.")
    if any(scale <= 0 for scale in scales):
        raise argparse.ArgumentTypeError("TTA scales must be positive integers.")
    return scales


def parse_member(value: str) -> EnsembleMember:
    parts = [part.strip() for part in value.split(":")]
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError(
            "Member must be model_name:checkpoint_prefix:weight[:checkpoint_suffix]."
        )

    model_name, checkpoint_prefix, weight_text = parts[:3]
    checkpoint_suffix = parts[3] if len(parts) == 4 else "best"

    if model_name not in SUPPORTED_MODELS:
        raise argparse.ArgumentTypeError(f"Unsupported model_name: {model_name}")

    try:
        weight = float(weight_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Member weight must be a float.") from exc

    if weight <= 0:
        raise argparse.ArgumentTypeError("Member weight must be positive.")

    return EnsembleMember(
        model_name=model_name,
        checkpoint_prefix=checkpoint_prefix,
        weight=weight,
        checkpoint_suffix=checkpoint_suffix,
    )


def load_fold_model(
    checkpoint_path: Path,
    model_name: str,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def build_loader(
    image_names: list[str],
    img_size: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    center_crop_resize: int | None,
) -> DataLoader:
    if center_crop_resize is None:
        transform = get_eval_transform(img_size)
    else:
        transform = get_eval_center_crop_transform(
            img_size=img_size,
            resize_size=center_crop_resize,
        )
    dataset = TestDataset(image_names, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--member",
        action="append",
        required=True,
        type=parse_member,
        help=(
            "Ensemble member in the form "
            "model_name:checkpoint_prefix:weight[:checkpoint_suffix]. "
            "Repeat this argument for multiple models."
        ),
    )
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--output", type=Path, default=RESULT_FILE)
    parser.add_argument("--no-tta", action="store_true")
    parser.add_argument(
        "--tta-scales",
        type=parse_tta_scales,
        default=None,
        help="Comma-separated eval image sizes. Defaults to --img-size only.",
    )
    parser.add_argument(
        "--center-crop-resize",
        type=int,
        default=None,
        help="Resize shorter side before CenterCrop(--img-size). Disabled by default.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_names = read_test_names()
    tta_scales = args.tta_scales or [args.img_size]

    total_weight = sum(member.weight for member in args.member)
    if total_weight <= 0:
        raise ValueError("Total ensemble weight must be positive.")

    loaded_members: list[tuple[EnsembleMember, list[torch.nn.Module]]] = []
    for member in args.member:
        models = []
        for fold in range(FOLDS):
            checkpoint_path = CHECKPOINT_DIR / (
                f"{member.checkpoint_prefix}_fold{fold}_{member.checkpoint_suffix}.pth"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
            models.append(load_fold_model(checkpoint_path, member.model_name, device))
        loaded_members.append((member, models))

    all_probs = torch.zeros(len(image_names), NUM_CLASSES, device=device)
    for img_size in tta_scales:
        loader = build_loader(
            image_names=image_names,
            img_size=img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            center_crop_resize=args.center_crop_resize,
        )

        offset = 0
        for images, _names in tqdm(loader, desc=f"ensemble infer {img_size}"):
            images = images.to(device, non_blocking=True)
            batch_probs = torch.zeros(images.size(0), NUM_CLASSES, device=device)

            for member, models in loaded_members:
                member_probs = torch.zeros(images.size(0), NUM_CLASSES, device=device)
                for model in models:
                    logits = model(images)
                    fold_probs = F.softmax(logits, dim=1)
                    if not args.no_tta:
                        flipped_logits = model(torch.flip(images, dims=[3]))
                        fold_probs = (
                            fold_probs + F.softmax(flipped_logits, dim=1)
                        ) / 2.0
                    member_probs += fold_probs
                member_probs /= len(models)
                batch_probs += member_probs * (member.weight / total_weight)

            batch_size = images.size(0)
            all_probs[offset : offset + batch_size] += batch_probs
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
