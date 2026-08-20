"""Lesson 39: normalize and freeze a verified real dataset into train/valid/test."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from captcha_break.data import IMAGE_SUFFIXES, label_from_filename
from captcha_break.dataset_split import (
    character_counts,
    copy_normalized_captcha,
    make_three_way_split,
    sha256_file,
)
from captcha_break.project_generator import PROJECT_ALPHABET
from captcha_break.real_analysis import classify_real_style, measure_real_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--train-size", type=int, default=349)
    parser.add_argument("--validation-size", type=int, default=75)
    parser.add_argument("--test-size", type=int, default=75)
    parser.add_argument("--seed", type=int, default=2039)
    parser.add_argument("--candidates", type=int, default=5000)
    return parser


def read_and_verify_export(source: Path) -> tuple[list[Path], dict[str, dict[str, str]]]:
    labels_path = source / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"labels.csv not found: {labels_path}")
    with labels_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["filename"]: row for row in csv.DictReader(stream)}
    paths = sorted(
        path
        for path in source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if set(rows) != {path.name for path in paths}:
        raise ValueError("labels.csv and exported image filenames do not match")
    for path in paths:
        row = rows[path.name]
        label = label_from_filename(path, PROJECT_ALPHABET)
        if row["label"] != label:
            raise ValueError(f"label mismatch in labels.csv: {path.name}")
        if row["sha256"].lower() != sha256_file(path):
            raise ValueError(f"source hash mismatch: {path.name}")
    return paths, rows


def write_report(
    output: Path,
    split_paths: dict[str, tuple[Path, ...]],
    styles: dict[Path, object],
) -> None:
    lines = [
        "Verified real CAPTCHA split",
        "Test is sealed: do not use it for training, generator tuning, or model selection.",
        "",
    ]
    for name, paths in split_paths.items():
        style_counts = Counter(str(styles[path]) for path in paths)
        counts = character_counts(paths, PROJECT_ALPHABET)
        lines.extend(
            [
                f"{name}: {len(paths)} images",
                f"  styles: {dict(style_counts)}",
                "  characters: "
                + ", ".join(f"{character}={counts[character]}" for character in PROJECT_ALPHABET),
                "",
            ]
        )
    (output / "split_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"source directory not found: {source}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    paths, rows = read_and_verify_export(source)
    styles = {path: classify_real_style(measure_real_image(path)) for path in paths}
    split = make_three_way_split(
        paths,
        styles,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        alphabet=PROJECT_ALPHABET,
        seed=args.seed,
        candidates=args.candidates,
    )
    split_paths = {
        "train": split.train,
        "validation": split.validation,
        "test": split.test,
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    normalization_counts: Counter[str] = Counter()
    for split_name, selected in split_paths.items():
        directory = output / split_name
        for path in selected:
            destination = directory / path.name
            normalization_counts[copy_normalized_captcha(path, destination)] += 1
            manifest_rows.append(
                {
                    "split": split_name,
                    "filename": path.name,
                    "label": rows[path.name]["label"],
                    "source_filename": rows[path.name]["source_filename"],
                    "source_sha256": rows[path.name]["sha256"],
                    "normalized_sha256": sha256_file(destination),
                    "style": str(styles[path]),
                }
            )
    with (output / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = [
            "split",
            "filename",
            "label",
            "source_filename",
            "source_sha256",
            "normalized_sha256",
            "style",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    write_report(output, split_paths, styles)
    print(f"Source preserved: {source}")
    print(f"Training: {output / 'train'} ({len(split.train)})")
    print(f"Validation: {output / 'validation'} ({len(split.validation)})")
    print(f"Sealed test: {output / 'test'} ({len(split.test)})")
    print(f"Normalization: {dict(normalization_counts)}")
    print(f"Manifest: {output / 'split_manifest.csv'}")
    print(f"Report: {output / 'split_report.txt'}")


if __name__ == "__main__":
    main()
