"""Lesson 19: render the same labels with all three visual profiles."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from captcha_break.project_generator import (
    VISUAL_STYLES,
    ProjectCaptchaGenerator,
    ProjectCaptchaStyle,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--font", default=None)
    parser.add_argument("--output", default="artifacts/multi_style_preview")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.rows <= 0:
        raise ValueError("rows must be positive")

    style = ProjectCaptchaStyle(font_path=args.font)
    generator = ProjectCaptchaGenerator(style)
    label_rng = random.Random(args.seed)
    labels = [
        "".join(label_rng.choice(style.alphabet) for _ in range(style.length))
        for _ in range(args.rows)
    ]

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_width = style.width + 16
    cell_height = style.height + 30
    sheet = Image.new("L", (len(VISUAL_STYLES) * cell_width, args.rows * cell_height), 255)
    draw = ImageDraw.Draw(sheet)
    caption_font = ImageFont.load_default()

    for row, label in enumerate(labels):
        for column, visual_style in enumerate(VISUAL_STYLES):
            image, _ = generator.generate(
                label,
                rng=random.Random(args.seed + row),
                visual_style=visual_style,
            )
            image.save(output_dir / f"{row + 1:02d}_{label}_{visual_style}.png")
            x = column * cell_width + 8
            y = row * cell_height + 4
            sheet.paste(image, (x, y))
            draw.text(
                (x, y + style.height + 4),
                f"{label} | {visual_style}",
                font=caption_font,
                fill=0,
            )

    preview_path = output_dir / "preview_grid.png"
    sheet.save(preview_path)
    print(f"Default weights: {style.style_weights}")
    print(f"Font: {generator.font_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
