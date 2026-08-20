"""Lesson 34: evaluate ddddocr as a local baseline on labeled real captchas."""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from captcha_break.data import IMAGE_SUFFIXES, label_from_filename
from captcha_break.ddddocr_adapter import DdddOcrRecognizer, OcrMetrics, calculate_ocr_metrics
from captcha_break.project_generator import PROJECT_ALPHABET
from captcha_break.real_analysis import classify_real_style, measure_real_image


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    filename: str
    target: str
    visual_style: str
    prediction: str
    milliseconds: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_directory", type=Path)
    parser.add_argument("--model", choices=("default", "beta", "both"), default="both")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/ddddocr_evaluation.csv"))
    return parser


def evaluate(
    paths: list[Path],
    recognizer: DdddOcrRecognizer,
) -> list[EvaluationRecord]:
    records: list[EvaluationRecord] = []
    for path in paths:
        target = label_from_filename(path, PROJECT_ALPHABET)
        visual_style = classify_real_style(measure_real_image(path))
        started = time.perf_counter()
        prediction = recognizer.predict(path.read_bytes())
        milliseconds = (time.perf_counter() - started) * 1000
        records.append(
            EvaluationRecord(
                filename=path.name,
                target=target,
                visual_style=visual_style,
                prediction=prediction,
                milliseconds=milliseconds,
            )
        )
    return records


def metrics_for(records: list[EvaluationRecord]) -> OcrMetrics:
    return calculate_ocr_metrics(
        [record.target for record in records],
        [record.prediction for record in records],
    )


def print_metrics(name: str, records: list[EvaluationRecord]) -> None:
    metrics = metrics_for(records)
    average_ms = sum(record.milliseconds for record in records) / len(records)
    print(
        f"{name}: char={metrics.character_accuracy:.2%}, "
        f"exact={metrics.exact_accuracy:.2%}, "
        f"length={metrics.matching_length_accuracy:.2%}, "
        f"average={average_ms:.2f} ms/image"
    )
    for visual_style in ("clean_outline", "noisy_outline", "solid"):
        subset = [record for record in records if record.visual_style == visual_style]
        if not subset:
            continue
        subset_metrics = metrics_for(subset)
        print(
            f"  {visual_style} ({len(subset)}): "
            f"char={subset_metrics.character_accuracy:.2%}, "
            f"exact={subset_metrics.exact_accuracy:.2%}"
        )
    errors = [
        f"{record.target}->{record.prediction}"
        for record in records
        if record.target != record.prediction
    ]
    print(f"  errors: {', '.join(errors) if errors else 'none'}")


def save_csv(
    output_path: Path,
    paths: list[Path],
    results: dict[str, list[EvaluationRecord]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records_by_model = {
        name: {record.filename: record for record in records}
        for name, records in results.items()
    }
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = ["filename", "target", "visual_style"]
        for name in results:
            fieldnames.extend((f"{name}_prediction", f"{name}_correct", f"{name}_ms"))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            first = next(records[path.name] for records in records_by_model.values())
            row: dict[str, str | bool] = {
                "filename": first.filename,
                "target": first.target,
                "visual_style": first.visual_style,
            }
            for name, records in records_by_model.items():
                record = records[path.name]
                row[f"{name}_prediction"] = record.prediction
                row[f"{name}_correct"] = record.prediction == record.target
                row[f"{name}_ms"] = f"{record.milliseconds:.3f}"
            writer.writerow(row)


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

    model_names = ("default", "beta") if args.model == "both" else (args.model,)
    print(f"ddddocr: {version('ddddocr')}")
    print(f"Images: {len(paths)}")
    print("Character range: unrestricted (set_ranges is intentionally not used)\n")

    results: dict[str, list[EvaluationRecord]] = {}
    for model_name in model_names:
        recognizer = DdddOcrRecognizer(beta=model_name == "beta")
        records = evaluate(paths, recognizer)
        results[model_name] = records
        print_metrics(model_name, records)
        print()

    output_path = args.output.expanduser().resolve()
    save_csv(output_path, paths, results)
    print(f"CSV: {output_path}")


if __name__ == "__main__":
    main()
