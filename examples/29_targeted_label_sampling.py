"""Lesson 29: preview targeted hard-character sampling before training v3."""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from captcha_break.data import TargetedLabelSampling
from captcha_break.project_generator import ProjectCaptchaGenerator, ProjectCaptchaStyle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--target-characters", default="JMPTUWV")
    parser.add_argument("--target-probability", type=float, default=0.35)
    parser.add_argument(
        "--target-position-weights",
        type=float,
        nargs=4,
        default=(1.0, 3.0, 1.0, 1.0),
        metavar=("P1", "P2", "P3", "P4"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/targeted_sampling"))
    return parser


def count_characters(labels: list[str]) -> list[Counter[str]]:
    return [Counter(label[position] for label in labels) for position in range(4)]


def build_report(
    uniform_labels: list[str], targeted_labels: list[str], sampling: TargetedLabelSampling
) -> str:
    uniform_counts = count_characters(uniform_labels)
    targeted_counts = count_characters(targeted_labels)
    target_set = set(sampling.characters)
    lines = [
        f"Labels in each experiment: {len(uniform_labels)}",
        f"Target characters: {sampling.characters}",
        f"Target replacement probability: {sampling.probability:.0%}",
        f"Position weights: {sampling.position_weights}",
        "",
        "Share occupied by target characters:",
    ]
    for position in range(4):
        uniform_target_count = sum(uniform_counts[position][char] for char in target_set)
        targeted_target_count = sum(targeted_counts[position][char] for char in target_set)
        lines.append(
            f"  position {position + 1}: uniform={uniform_target_count / len(uniform_labels):.2%}, "
            f"targeted={targeted_target_count / len(targeted_labels):.2%}"
        )

    lines.extend(["", "Targeted character counts by position:"])
    for character in sampling.characters:
        counts = [targeted_counts[position][character] for position in range(4)]
        lines.append(f"  {character}: {counts}")
    lines.extend(
        [
            "",
            "Important:",
            "  Use targeted sampling only for the training dataset.",
            "  Keep synthetic validation uniform and real validation unchanged.",
        ]
    )
    return "\n".join(lines)


def save_preview(
    labels: list[str], generator: ProjectCaptchaGenerator, path: Path, seed: int
) -> None:
    columns, rows = 3, 4
    cell_width, cell_height = 216, 78
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, label in enumerate(labels[: columns * rows]):
        image, _ = generator.generate(text=label, rng=random.Random(seed + index))
        x = index % columns * cell_width + 8
        y = index // columns * cell_height + 4
        sheet.paste(image.convert("RGB"), (x, y))
        draw.text((x, y + 54), label, fill="black", font=font)
    sheet.save(path)


def main() -> None:
    args = build_parser().parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    style = ProjectCaptchaStyle()
    sampling = TargetedLabelSampling(
        characters=args.target_characters.upper(),
        probability=args.target_probability,
        position_weights=tuple(args.target_position_weights),
    )
    sampling.validate_for(style.alphabet, style.length)
    uniform_rng = random.Random(args.seed)
    targeted_rng = random.Random(args.seed)
    uniform_labels = [
        "".join(uniform_rng.choice(style.alphabet) for _ in range(style.length))
        for _ in range(args.samples)
    ]
    targeted_labels = [
        sampling.sample(style.alphabet, style.length, targeted_rng) for _ in range(args.samples)
    ]

    output_directory = args.output.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "distribution.txt"
    preview_path = output_directory / "preview.png"
    report = build_report(uniform_labels, targeted_labels, sampling)
    report_path.write_text(report + "\n", encoding="utf-8")
    save_preview(targeted_labels, ProjectCaptchaGenerator(style), preview_path, args.seed)
    print(report)
    print(f"\nReport: {report_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
