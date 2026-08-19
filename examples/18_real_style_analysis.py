"""Lesson 18: measure labels, geometry, darkness, and styles in real captchas."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from captcha_break.project_generator import PROJECT_ALPHABET
from captcha_break.real_analysis import classify_real_style, measure_real_image

IMAGE_SUFFIXES = {".bmp", ".jfif", ".jpeg", ".jpg", ".png"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="directory containing labeled images")
    return parser


def label_from_path(path: Path) -> str:
    """Read the four-character label before the first underscore."""

    label = path.stem.split("_", maxsplit=1)[0].upper()
    if len(label) != 4 or not set(label) <= set(PROJECT_ALPHABET):
        raise ValueError(f"invalid four-character label in filename: {path.name}")
    return label


def print_range(name: str, values: list[float]) -> None:
    array = np.asarray(values)
    print(
        f"{name:>18}: min={array.min():.4f}, median={np.median(array):.4f}, max={array.max():.4f}"
    )


def main() -> None:
    input_dir = build_parser().parse_args().input.expanduser().resolve()
    paths = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no captcha images found in {input_dir}")

    metrics = [(label_from_path(path), measure_real_image(path)) for path in paths]
    sizes: Counter[tuple[int, int]] = Counter()
    for path in paths:
        with Image.open(path) as image:
            sizes[image.size] += 1
    characters = Counter("".join(label for label, _ in metrics))
    missing = "".join(character for character in PROJECT_ALPHABET if not characters[character])
    styles = Counter(classify_real_style(item) for _, item in metrics)

    print(f"Directory: {input_dir}")
    print(f"Images: {len(metrics)}")
    print(f"Sizes: {dict(sizes)}")
    print(f"Covered characters: {len(characters)}/{len(PROJECT_ALPHABET)}")
    print(f"Missing characters: {missing or '(none)'}")
    print(f"Estimated styles: {dict(styles)}")
    print_range("mean gray", [item.mean_gray for _, item in metrics])
    print_range("dark ratio", [item.dark_ratio for _, item in metrics])
    print_range("foreground ratio", [item.foreground_ratio for _, item in metrics])

    width, height = next(iter(sizes))
    print(f"Touch left: {sum(item.bbox[0] == 0 for _, item in metrics)}/{len(metrics)}")
    print(f"Touch top: {sum(item.bbox[1] == 0 for _, item in metrics)}/{len(metrics)}")
    print(f"Touch right: {sum(item.bbox[2] == width for _, item in metrics)}/{len(metrics)}")
    print(f"Touch bottom: {sum(item.bbox[3] == height for _, item in metrics)}/{len(metrics)}")


if __name__ == "__main__":
    main()
