"""Lesson 26: compare real and synthetic noisy-outline captchas with matching labels."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from captcha_break.data import IMAGE_SUFFIXES, label_from_filename
from captcha_break.project_generator import PROJECT_ALPHABET, ProjectCaptchaGenerator
from captcha_break.real_analysis import (
    RealImageMetrics,
    classify_real_style,
    measure_image,
    measure_real_image,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_directory", type=Path)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default="artifacts/noisy_style_calibration")
    return parser


def print_statistics(name: str, metrics: list[RealImageMetrics]) -> None:
    values = np.asarray(
        [(item.mean_gray, item.dark_ratio, item.foreground_ratio) for item in metrics]
    )
    print(f"{name} (n={len(metrics)})")
    for column, label in enumerate(("mean gray", "dark ratio", "foreground ratio")):
        print(
            f"  {label:16s} median={np.median(values[:, column]):.4f}, "
            f"mean={values[:, column].mean():.4f}"
        )


def main() -> None:
    args = build_parser().parse_args()
    real_directory = args.real_directory.expanduser().resolve()
    noisy_paths = [
        path
        for path in sorted(real_directory.iterdir())
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and classify_real_style(measure_real_image(path)) == "noisy_outline"
    ]
    if not noisy_paths:
        raise FileNotFoundError(f"no noisy-outline captchas found in {real_directory}")

    generator = ProjectCaptchaGenerator()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_width = generator.style.width + 16
    cell_height = generator.style.height + 30
    sheet = Image.new("RGB", (2 * cell_width, len(noisy_paths) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    real_metrics: list[RealImageMetrics] = []
    synthetic_metrics: list[RealImageMetrics] = []

    for row, path in enumerate(noisy_paths):
        label = label_from_filename(path, PROJECT_ALPHABET)
        with Image.open(path) as source:
            real_image = source.convert("RGB")
        synthetic_image, _ = generator.generate(
            label,
            rng=random.Random(args.seed + row),
            visual_style="noisy_outline",
        )
        real_metrics.append(measure_image(real_image, name=path.name))
        synthetic_metrics.append(measure_image(synthetic_image, name=label))
        synthetic_image.save(output_dir / f"{label}_synthetic.png")

        for column, (source_name, image) in enumerate(
            (("real", real_image), ("synthetic", synthetic_image.convert("RGB")))
        ):
            x = column * cell_width + 8
            y = row * cell_height + 4
            sheet.paste(image, (x, y))
            draw.text(
                (x, y + generator.style.height + 4),
                f"{source_name}: {label}",
                fill="black",
                font=font,
            )

    preview_path = output_dir / "preview_grid.png"
    sheet.save(preview_path)
    print_statistics("Real noisy outline", real_metrics)
    print_statistics("Synthetic noisy outline", synthetic_metrics)
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
