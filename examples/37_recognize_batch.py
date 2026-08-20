"""Lesson 37: batch-recognize a local directory and save an auditable CSV report."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from captcha_break.data import IMAGE_SUFFIXES, label_from_filename
from captcha_break.ddddocr_adapter import calculate_ocr_metrics
from captcha_break.project_generator import PROJECT_ALPHABET
from captcha_break.recognizer import BatchRecognitionItem, ProjectCaptchaRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--labeled", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/batch_recognition.csv"),
    )
    return parser


def save_csv(
    output_path: Path,
    items: list[BatchRecognitionItem],
    targets: dict[str, str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "filename",
        "target",
        "prediction",
        "correct",
        "succeeded",
        "status",
        "model_used",
        "used_fallback",
        "beta_text",
        "default_text",
        "fallback_reason",
        "milliseconds",
        "error",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            result = item.result
            target = targets.get(item.identifier, "")
            prediction = result.text if result is not None else ""
            writer.writerow(
                {
                    "filename": item.identifier,
                    "target": target,
                    "prediction": prediction,
                    "correct": prediction == target if target else "",
                    "succeeded": item.succeeded,
                    "status": result.status if result is not None else "failed",
                    "model_used": result.model_used if result is not None else "",
                    "used_fallback": result.used_fallback if result is not None else "",
                    "beta_text": result.beta_text if result is not None else "",
                    "default_text": (result.default_text or "") if result is not None else "",
                    "fallback_reason": (
                        result.fallback_reason or "" if result is not None else ""
                    ),
                    "milliseconds": (
                        f"{result.milliseconds:.3f}" if result is not None else ""
                    ),
                    "error": item.error or "",
                }
            )


def main() -> None:
    args = build_parser().parse_args()
    image_directory = args.image_directory.expanduser().resolve()
    if not image_directory.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {image_directory}")

    paths = sorted(
        path
        for path in image_directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"no supported captcha images found in {image_directory}")

    recognizer = ProjectCaptchaRecognizer()
    items = recognizer.recognize_many((path.name, path) for path in paths)
    successful = [item for item in items if item.result is not None]
    failures = [item for item in items if item.result is None]
    fallback_count = sum(item.result.used_fallback for item in successful)
    review_count = sum(not item.result.is_structurally_valid for item in successful)
    average_ms = (
        sum(item.result.milliseconds for item in successful) / len(successful)
        if successful
        else 0.0
    )

    print(f"Images: {len(items)}")
    print(f"Succeeded: {len(successful)}; failed: {len(failures)}")
    print(f"Fallbacks: {fallback_count}; structurally invalid: {review_count}")
    print(f"Average inference time: {average_ms:.2f} ms/image")

    targets: dict[str, str] = {}
    if args.labeled:
        targets = {
            path.name: label_from_filename(path, PROJECT_ALPHABET) for path in paths
        }
        predictions = [
            item.result.text if item.result is not None else "" for item in items
        ]
        metrics = calculate_ocr_metrics(
            [targets[item.identifier] for item in items],
            predictions,
        )
        print(
            f"Labeled metrics: char={metrics.character_accuracy:.2%}, "
            f"exact={metrics.exact_accuracy:.2%}, "
            f"length={metrics.matching_length_accuracy:.2%}"
        )

    output_path = args.output.expanduser().resolve()
    save_csv(output_path, items, targets)
    print(f"CSV: {output_path}")


if __name__ == "__main__":
    main()
