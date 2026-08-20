"""Lesson 35: evaluate Beta-first ddddocr with a lazy Default fallback."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from captcha_break.data import IMAGE_SUFFIXES, label_from_filename
from captcha_break.ddddocr_adapter import (
    DdddOcrFallbackRecognizer,
    FallbackPrediction,
    calculate_ocr_metrics,
)
from captcha_break.project_generator import PROJECT_ALPHABET


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    filename: str
    target: str
    result: FallbackPrediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/ddddocr_fallback.csv"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_directory = args.image_directory.expanduser().resolve()
    if not image_directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_directory}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")

    paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.limit is not None:
        paths = paths[: args.limit]
    if not paths:
        raise FileNotFoundError(f"no supported captcha images found in {image_directory}")

    recognizer = DdddOcrFallbackRecognizer(
        expected_length=4,
        alphabet=PROJECT_ALPHABET,
    )
    records = [
        EvaluationRecord(
            filename=path.name,
            target=label_from_filename(path, PROJECT_ALPHABET),
            result=recognizer.predict_detailed(path.read_bytes()),
        )
        for path in paths
    ]

    metrics = calculate_ocr_metrics(
        [record.target for record in records],
        [record.result.text for record in records],
    )
    fallback_count = sum(record.result.used_fallback for record in records)
    default_count = sum(record.result.model_used == "default" for record in records)
    invalid_count = sum(not record.result.is_valid for record in records)
    average_ms = sum(record.result.milliseconds for record in records) / len(records)

    print(f"Images: {len(records)}")
    print(f"Alphabet: {PROJECT_ALPHABET}")
    print("Rule: Beta first; Default only when Beta is not a valid 4-character result")
    print(
        f"combined: char={metrics.character_accuracy:.2%}, "
        f"exact={metrics.exact_accuracy:.2%}, "
        f"length={metrics.matching_length_accuracy:.2%}, "
        f"average={average_ms:.2f} ms/image"
    )
    print(
        f"Fallback attempted: {fallback_count}; Default selected: {default_count}; "
        f"invalid after fallback: {invalid_count}"
    )
    errors = [
        f"{record.target}->{record.result.text} ({record.result.model_used})"
        for record in records
        if record.target != record.result.text
    ]
    print(f"Errors: {', '.join(errors) if errors else 'none'}")

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = [
            "filename",
            "target",
            "prediction",
            "correct",
            "model_used",
            "beta_prediction",
            "default_prediction",
            "fallback_reason",
            "valid",
            "milliseconds",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            result = record.result
            writer.writerow(
                {
                    "filename": record.filename,
                    "target": record.target,
                    "prediction": result.text,
                    "correct": result.text == record.target,
                    "model_used": result.model_used,
                    "beta_prediction": result.beta_text,
                    "default_prediction": result.default_text or "",
                    "fallback_reason": result.fallback_reason or "",
                    "valid": result.is_valid,
                    "milliseconds": f"{result.milliseconds:.3f}",
                }
            )
    print(f"CSV: {output_path}")


if __name__ == "__main__":
    main()
