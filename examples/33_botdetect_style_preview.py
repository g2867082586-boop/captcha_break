"""Lesson 33: compare real captchas with BotDetect-inspired synthetic matches."""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from captcha_break.project_generator import (
    BOTDETECT_STYLE_NAMES,
    ProjectCaptchaGenerator,
    VisualStyle,
    project_style_for_source,
)
from captcha_break.real_analysis import classify_real_style, measure_real_image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".jfif", ".png", ".bmp"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_directory", type=Path)
    parser.add_argument("--per-style", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2033)
    parser.add_argument("--output", default="artifacts/botdetect_style_preview")
    return parser


def label_from_path(path: Path) -> str:
    return path.stem.split("_", maxsplit=1)[0].upper()


def collect_examples(directory: Path, per_style: int, seed: int) -> list[tuple[Path, VisualStyle]]:
    grouped: dict[VisualStyle, list[Path]] = defaultdict(list)
    paths = sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    for path in paths:
        style = classify_real_style(measure_real_image(path))
        grouped[style].append(path)

    rng = random.Random(seed)
    selected: list[tuple[Path, VisualStyle]] = []
    for style in ("clean_outline", "noisy_outline", "solid"):
        candidates = grouped[style]
        rng.shuffle(candidates)
        selected.extend((path, style) for path in candidates[:per_style])
    return selected


def main() -> None:
    args = build_parser().parse_args()
    if args.per_style <= 0:
        raise ValueError("per-style must be positive")
    real_directory = args.real_directory.expanduser().resolve()
    if not real_directory.is_dir():
        raise FileNotFoundError(f"real image directory does not exist: {real_directory}")

    selected = collect_examples(real_directory, args.per_style, args.seed)
    if not selected:
        raise ValueError(f"no supported images found in {real_directory}")

    style = project_style_for_source("botdetect")
    generator = ProjectCaptchaGenerator(style)
    rng = random.Random(args.seed)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_width = style.width + 16
    cell_height = style.height + 29
    sheet = Image.new("L", (cell_width * 2, cell_height * len(selected)), 255)
    draw = ImageDraw.Draw(sheet)
    caption_font = ImageFont.load_default()

    for row, (real_path, visual_style) in enumerate(selected):
        label = label_from_path(real_path)
        with Image.open(real_path) as source:
            real_image = source.convert("L")
        synthetic, _ = generator.generate(label, rng=rng, visual_style=visual_style)
        synthetic.save(output_dir / f"{label}_{visual_style}.png")

        y = row * cell_height + 3
        sheet.paste(real_image, (8, y))
        sheet.paste(synthetic, (cell_width + 8, y))
        reference_name = BOTDETECT_STYLE_NAMES[visual_style]
        draw.text(
            (8, y + style.height + 3),
            f"real {label} | {reference_name}",
            font=caption_font,
            fill=0,
        )
        draw.text(
            (cell_width + 8, y + style.height + 3),
            f"synthetic {label} | {reference_name}",
            font=caption_font,
            fill=0,
        )

    preview_path = output_dir / "preview_grid.png"
    sheet.save(preview_path)
    print(f"Compared samples: {len(selected)}")
    print(f"Resolved font pool ({len(generator.font_paths)}):")
    for font_path in generator.font_paths:
        print(f"  {font_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
