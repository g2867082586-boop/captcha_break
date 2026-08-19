"""Lesson 31: stratify real captchas into copied training and validation folders."""

from __future__ import annotations

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from captcha_break.data import IMAGE_SUFFIXES, label_from_filename
from captcha_break.project_generator import PROJECT_ALPHABET, VISUAL_STYLES, VisualStyle
from captcha_break.real_analysis import classify_real_style, measure_real_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2031)
    parser.add_argument("--candidates", type=int, default=5000)
    return parser


def proportional_quotas(
    style_counts: Counter[VisualStyle], validation_size: int, total: int
) -> dict[VisualStyle, int]:
    raw = {
        style: style_counts[style] * validation_size / total for style in VISUAL_STYLES
    }
    quotas = {style: int(raw[style]) for style in VISUAL_STYLES}
    remaining = validation_size - sum(quotas.values())
    ranked = sorted(VISUAL_STYLES, key=lambda style: raw[style] - quotas[style], reverse=True)
    for style in ranked[:remaining]:
        quotas[style] += 1
    return quotas


def character_counts(paths: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        counts.update(label_from_filename(path, PROJECT_ALPHABET))
    return counts


def choose_validation(
    groups: dict[VisualStyle, list[Path]],
    quotas: dict[VisualStyle, int],
    full_character_counts: Counter[str],
    validation_size: int,
    total: int,
    seed: int,
    candidates: int,
) -> list[Path]:
    best_paths: list[Path] = []
    best_score: tuple[int, int, int, float] | None = None
    expected = {
        character: full_character_counts[character] * validation_size / total
        for character in PROJECT_ALPHABET
    }

    for attempt in range(candidates):
        selected: list[Path] = []
        for style_index, style in enumerate(VISUAL_STYLES):
            candidates_for_style = groups[style].copy()
            random.Random(seed + attempt * len(VISUAL_STYLES) + style_index).shuffle(
                candidates_for_style
            )
            selected.extend(candidates_for_style[: quotas[style]])

        counts = character_counts(selected)
        covered = sum(counts[character] > 0 for character in PROJECT_ALPHABET)
        minimum = min(counts[character] for character in PROJECT_ALPHABET)
        capped_balance = sum(min(counts[character], 5) for character in PROJECT_ALPHABET)
        distribution_error = sum(
            abs(counts[character] - expected[character]) for character in PROJECT_ALPHABET
        )
        score = (covered, minimum, capped_balance, -distribution_error)
        if best_score is None or score > best_score:
            best_score = score
            best_paths = sorted(selected)
    return best_paths


def write_report(
    path: Path,
    train_paths: list[Path],
    validation_paths: list[Path],
    styles: dict[Path, VisualStyle],
) -> None:
    train_styles = Counter(styles[item] for item in train_paths)
    validation_styles = Counter(styles[item] for item in validation_paths)
    train_characters = character_counts(train_paths)
    validation_characters = character_counts(validation_paths)
    lines = [
        f"Training images: {len(train_paths)}",
        f"Validation images: {len(validation_paths)}",
        f"Training styles: {dict(train_styles)}",
        f"Validation styles: {dict(validation_styles)}",
        "",
        "Character counts (train / validation):",
    ]
    lines.extend(
        f"  {character}: {train_characters[character]} / {validation_characters[character]}"
        for character in PROJECT_ALPHABET
    )
    lines.extend(["", "Validation filenames:"])
    lines.extend(f"  {item.name}" for item in validation_paths)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")
    if args.train_size <= 0 or args.candidates <= 0:
        raise ValueError("train size and candidates must be positive")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    paths = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.train_size >= len(paths):
        raise ValueError("train size must leave at least one validation image")
    validation_size = len(paths) - args.train_size

    styles = {path: classify_real_style(measure_real_image(path)) for path in paths}
    groups: dict[VisualStyle, list[Path]] = defaultdict(list)
    for path in paths:
        label_from_filename(path, PROJECT_ALPHABET)
        groups[styles[path]].append(path)
    style_counts = Counter(styles.values())
    quotas = proportional_quotas(style_counts, validation_size, len(paths))
    validation_paths = choose_validation(
        groups,
        quotas,
        character_counts(paths),
        validation_size,
        len(paths),
        args.seed,
        args.candidates,
    )
    validation_set = set(validation_paths)
    train_paths = [path for path in paths if path not in validation_set]
    if len(train_paths) != args.train_size or len(validation_paths) != validation_size:
        raise RuntimeError("split sizes do not match the requested sizes")

    train_directory = output / "train"
    validation_directory = output / "validation"
    train_directory.mkdir(parents=True, exist_ok=False)
    validation_directory.mkdir(parents=True, exist_ok=False)
    for path in train_paths:
        shutil.copy2(path, train_directory / path.name)
    for path in validation_paths:
        shutil.copy2(path, validation_directory / path.name)

    report_path = output / "split_report.txt"
    write_report(report_path, train_paths, validation_paths, styles)
    print(f"Source preserved: {source}")
    print(f"Training: {train_directory} ({len(train_paths)})")
    print(f"Validation: {validation_directory} ({len(validation_paths)})")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
