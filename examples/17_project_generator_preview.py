"""Generate a contact sheet for tuning the project-specific captcha style."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from captcha_break.project_generator import ProjectCaptchaGenerator, ProjectCaptchaStyle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/project_generator_preview")
    parser.add_argument("--font", default=None)
    parser.add_argument("--text", default=None, help="fixed four-character preview label")
    parser.add_argument("--font-size", type=int, default=60)
    parser.add_argument("--horizontal-scale", type=float, default=1.35)
    parser.add_argument("--rotation", type=float, default=7.0)
    parser.add_argument("--overlap-min", type=int, default=5)
    parser.add_argument("--overlap-max", type=int, default=12)
    parser.add_argument("--background-noise", type=float, default=0.001)
    parser.add_argument("--foreground-noise", type=float, default=0.0)
    parser.add_argument("--scratch-count", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")

    style = ProjectCaptchaStyle(
        font_path=args.font,
        font_size=args.font_size,
        horizontal_scale=args.horizontal_scale,
        rotation_degrees=args.rotation,
        overlap_min=args.overlap_min,
        overlap_max=args.overlap_max,
        background_noise_density=args.background_noise,
        foreground_noise_density=args.foreground_noise,
        scratch_count=args.scratch_count,
    )
    generator = ProjectCaptchaGenerator(style)
    rng = random.Random(args.seed)
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    samples: list[tuple[Image.Image, str]] = []
    for index in range(args.count):
        requested_text = args.text if args.text is not None else ("KJUU" if index == 0 else None)
        image, label = generator.generate(requested_text, rng=rng)
        image.save(output_dir / f"{label}_{index + 1:03d}.png")
        samples.append((image, label))

    columns = min(4, len(samples))
    rows = (len(samples) + columns - 1) // columns
    cell_width = style.width + 16
    cell_height = style.height + 28
    sheet = Image.new("L", (columns * cell_width, rows * cell_height), 255)
    draw = ImageDraw.Draw(sheet)
    label_font = ImageFont.load_default()
    for index, (image, label) in enumerate(samples):
        column = index % columns
        row = index // columns
        x = column * cell_width + 8
        y = row * cell_height + 4
        sheet.paste(image, (x, y))
        draw.text((x, y + style.height + 4), label, font=label_font, fill=0)

    sheet_path = output_dir / "preview_grid.png"
    sheet.save(sheet_path)
    print(f"Font: {generator.font_path}")
    print(f"Samples: {output_dir}")
    print(f"Preview: {sheet_path}")


if __name__ == "__main__":
    main()
