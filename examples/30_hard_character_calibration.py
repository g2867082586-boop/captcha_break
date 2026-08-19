"""Lesson 30: compare difficult real characters with matched synthetic captchas."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from captcha_break.project_generator import PROJECT_ALPHABET, ProjectCaptchaGenerator
from captcha_break.real_analysis import classify_real_style, measure_real_image


@dataclass(frozen=True, slots=True)
class HardCase:
    filename: str
    target: str
    prediction: str
    position: int
    character: str
    confidence: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_directory", type=Path)
    parser.add_argument(
        "comparison_csv",
        type=Path,
        nargs="?",
        default=Path("artifacts/checkpoint_comparison_v2_v3/comparison.csv"),
    )
    parser.add_argument("--characters", default="JB3W")
    parser.add_argument("--per-character", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2030)
    parser.add_argument("--output", default="artifacts/hard_character_calibration")
    return parser


def load_hard_cases(path: Path, characters: str) -> list[HardCase]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))

    cases: list[HardCase] = []
    for row in rows:
        target = row["target"]
        prediction = row["second_prediction"]
        for position, character in enumerate(target):
            if character not in characters or prediction[position] == character:
                continue
            cases.append(
                HardCase(
                    filename=row["filename"],
                    target=target,
                    prediction=prediction,
                    position=position,
                    character=character,
                    confidence=float(row[f"second_confidence_{position + 1}"]),
                )
            )
    return cases


def select_cases(cases: list[HardCase], characters: str, limit: int) -> list[HardCase]:
    grouped: dict[str, list[HardCase]] = defaultdict(list)
    for case in cases:
        grouped[case.character].append(case)

    selected: list[HardCase] = []
    for character in characters:
        ranked = sorted(grouped[character], key=lambda case: case.confidence, reverse=True)
        selected.extend(ranked[:limit])
    return selected


def main() -> None:
    args = build_parser().parse_args()
    characters = args.characters.upper()
    if not characters or len(set(characters)) != len(characters):
        raise ValueError("characters must be non-empty and unique")
    invalid = sorted(set(characters) - set(PROJECT_ALPHABET))
    if invalid:
        raise ValueError(f"characters are outside the project alphabet: {invalid}")
    if args.per_character <= 0:
        raise ValueError("per-character must be positive")

    real_directory = args.real_directory.expanduser().resolve()
    all_cases = load_hard_cases(args.comparison_csv, characters)
    selected = select_cases(all_cases, characters, args.per_character)
    if not selected:
        raise ValueError("no matching v3 errors were found")

    generator = ProjectCaptchaGenerator()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_width = generator.style.width + 16
    cell_height = generator.style.height + 34
    sheet = Image.new("RGB", (2 * cell_width, len(selected) * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()

    for row, case in enumerate(selected):
        real_path = real_directory / case.filename
        if not real_path.is_file():
            raise FileNotFoundError(f"real image not found: {real_path}")
        visual_style = classify_real_style(measure_real_image(real_path))
        with Image.open(real_path) as source:
            real_image = source.convert("RGB")
        synthetic_image, _ = generator.generate(
            case.target,
            rng=random.Random(args.seed + row),
            visual_style=visual_style,
        )
        synthetic_rgb = synthetic_image.convert("RGB")
        synthetic_rgb.save(
            output_dir
            / f"{Path(case.filename).stem}_p{case.position + 1}_{case.character}_synthetic.png"
        )

        caption = (
            f"{case.character}@{case.position + 1}: "
            f"{case.target}->{case.prediction} {case.confidence:.0%} {visual_style}"
        )
        for column, (name, image) in enumerate(
            (("real", real_image), ("synthetic", synthetic_rgb))
        ):
            x = column * cell_width + 8
            y = row * cell_height + 4
            sheet.paste(image, (x, y))
            draw.text(
                (x, y + generator.style.height + 4),
                f"{name}: {caption}",
                fill="black",
                font=font,
            )

    preview_path = output_dir / "preview_grid.png"
    sheet.save(preview_path)
    counts = Counter(case.character for case in all_cases)
    print(f"All v3 errors for {characters}: {dict(counts)}")
    print(f"Selected: {len(selected)} highest-confidence errors")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
