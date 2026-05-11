from __future__ import annotations

import argparse
from pathlib import Path

from config import NUM_CLASSES, RESULT_FILE
from dataset import read_test_names


def validate_submission(path: Path) -> None:
    expected_names = read_test_names()
    lines = path.read_text(encoding="utf-8").splitlines()

    errors = []
    if len(lines) != len(expected_names):
        errors.append(f"line count {len(lines)} != expected {len(expected_names)}")

    for idx, line in enumerate(lines):
        parts = line.split("\t")
        if len(parts) != 2:
            errors.append(f"line {idx + 1}: expected two tab-separated columns")
            continue
        image_name, label_text = parts[0], parts[1].strip()
        if not image_name or image_name.strip() != image_name:
            errors.append(f"line {idx + 1}: image name has invalid whitespace")
            continue
        if idx < len(expected_names) and image_name != expected_names[idx]:
            errors.append(
                f"line {idx + 1}: image {image_name} != expected {expected_names[idx]}"
            )
        if not label_text.isdigit():
            errors.append(f"line {idx + 1}: label is not an integer")
            continue
        label = int(label_text)
        if label < 0 or label >= NUM_CLASSES:
            errors.append(f"line {idx + 1}: label {label} not in 0..{NUM_CLASSES - 1}")

    if errors:
        raise SystemExit("Submission validation failed:\n" + "\n".join(errors[:20]))
    print(f"Submission OK: {path}, {len(lines)} lines.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=RESULT_FILE)
    args = parser.parse_args()
    validate_submission(args.path)


if __name__ == "__main__":
    main()
