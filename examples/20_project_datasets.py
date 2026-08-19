"""Lesson 20: load synthetic training data and real validation data together."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from captcha_break.codec import decode_indices
from captcha_break.data import ProjectCaptchaDataset, RealCaptchaDataset
from captcha_break.project_generator import PROJECT_ALPHABET


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_directory", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="artifacts/dataset_preview.png")
    return parser


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Undo the 0..1 normalization for display only."""

    pixels = tensor.squeeze(0).mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
    return Image.fromarray(pixels)


def print_batch(
    name: str,
    images: torch.Tensor,
    targets: torch.Tensor,
    characters: str,
) -> list[str]:
    labels = [decode_indices(target.tolist(), characters) for target in targets]
    print(f"\n{name}")
    print(f"  image shape: {tuple(images.shape)}")
    print(f"  target shape: {tuple(targets.shape)}")
    print(f"  image dtype: {images.dtype}")
    print(f"  target dtype: {targets.dtype}")
    print(f"  image range: {images.min().item():.4f} .. {images.max().item():.4f}")
    print(f"  labels: {labels}")
    return labels


def save_comparison(
    synthetic_images: torch.Tensor,
    synthetic_labels: list[str],
    real_images: torch.Tensor,
    real_labels: list[str],
    output: Path,
) -> None:
    columns = min(4, len(synthetic_labels), len(real_labels))
    image_width = synthetic_images.shape[-1]
    image_height = synthetic_images.shape[-2]
    cell_width = image_width + 16
    cell_height = image_height + 30
    sheet = Image.new("L", (columns * cell_width, 2 * cell_height), 255)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    rows = (
        ("synthetic", synthetic_images, synthetic_labels),
        ("real", real_images, real_labels),
    )
    for row, (source_name, images, labels) in enumerate(rows):
        for column in range(columns):
            x = column * cell_width + 8
            y = row * cell_height + 4
            sheet.paste(tensor_to_image(images[column]), (x, y))
            draw.text(
                (x, y + image_height + 4),
                f"{source_name}: {labels[column]}",
                font=font,
                fill=0,
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.synthetic_size <= 0:
        raise ValueError("batch size and synthetic size must be positive")

    synthetic_dataset = ProjectCaptchaDataset(size=args.synthetic_size, seed=args.seed)
    real_dataset = RealCaptchaDataset(
        args.real_directory,
        characters=PROJECT_ALPHABET,
        length=4,
        expected_size=(200, 50),
    )
    synthetic_loader = DataLoader(
        synthetic_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    real_loader = DataLoader(
        real_dataset,
        batch_size=min(args.batch_size, len(real_dataset)),
        shuffle=False,
        num_workers=0,
    )

    synthetic_images, synthetic_targets = next(iter(synthetic_loader))
    real_images, real_targets = next(iter(real_loader))
    synthetic_labels = print_batch(
        "Synthetic preview batch (fixed seed)",
        synthetic_images,
        synthetic_targets,
        synthetic_dataset.characters,
    )
    real_labels = print_batch(
        "Real validation batch",
        real_images,
        real_targets,
        real_dataset.characters,
    )

    output = Path(args.output).expanduser().resolve()
    save_comparison(
        synthetic_images,
        synthetic_labels,
        real_images,
        real_labels,
        output,
    )
    print(f"\nSynthetic samples per epoch: {len(synthetic_dataset)}")
    print(f"Real validation samples: {len(real_dataset)}")
    print(f"Preview: {output}")


if __name__ == "__main__":
    main()
