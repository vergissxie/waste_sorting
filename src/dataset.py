from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from config import (
    EVAL_RESIZE_RATIO,
    GARBAGE_DICT_FILE,
    IMAGE_EXTENSIONS,
    IMG_SIZE,
    NUM_CLASSES,
    TEST_DIR,
    TEST_PATH_FILE,
    TRAIN_DIR,
)


ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_samples(train_dir: Path = TRAIN_DIR) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label in range(NUM_CLASSES):
        class_dir = train_dir / str(label)
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    return samples


def read_test_names(test_path_file: Path = TEST_PATH_FILE) -> list[str]:
    return [
        line.strip()
        for line in test_path_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def audit_data() -> dict[str, object]:
    train_counts = {}
    missing_train_dirs = []
    for label in range(NUM_CLASSES):
        class_dir = TRAIN_DIR / str(label)
        if not class_dir.exists():
            missing_train_dirs.append(str(label))
            train_counts[str(label)] = 0
            continue
        train_counts[str(label)] = sum(
            1
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    test_names = read_test_names()
    missing_test_images = [
        name for name in test_names if not (TEST_DIR / name).is_file()
    ]

    label_keys_ok = False
    if GARBAGE_DICT_FILE.exists():
        label_map = json.loads(GARBAGE_DICT_FILE.read_text(encoding="utf-8"))
        label_keys_ok = sorted(label_map.keys(), key=int) == [
            str(i) for i in range(NUM_CLASSES)
        ]

    return {
        "train_total": sum(train_counts.values()),
        "train_counts": train_counts,
        "missing_train_dirs": missing_train_dirs,
        "test_count": len(test_names),
        "missing_test_images": missing_test_images,
        "label_keys_ok": label_keys_ok,
    }


def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def get_eval_transform(
    img_size: int = IMG_SIZE,
    resize_ratio: float = EVAL_RESIZE_RATIO,
    mode: str = "crop",
) -> transforms.Compose:
    if mode == "resize":
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    if mode != "crop":
        raise ValueError(f"Unsupported eval transform mode: {mode}")

    resize_size = max(img_size, round(img_size * resize_ratio))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_center_crop_transform(
    img_size: int = IMG_SIZE,
    resize_size: int = 292,
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class GarbageDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, int]],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TestDataset(Dataset):
    def __init__(
        self,
        image_names: list[str],
        test_dir: Path = TEST_DIR,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.image_names = image_names
        self.test_dir = test_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_name = self.image_names[index]
        image = Image.open(self.test_dir / image_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, image_name

