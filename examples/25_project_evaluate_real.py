"""Lesson 25: evaluate a project checkpoint on every labeled real captcha."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from captcha_break.codec import decode_indices
from captcha_break.data import RealCaptchaDataset
from captcha_break.models import ProjectCaptchaCNN
from captcha_break.project_generator import VISUAL_STYLES, ProjectCaptchaStyle, VisualStyle
from captcha_break.real_analysis import classify_real_style, measure_real_image
from captcha_break.training import resolve_device


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    filename: str
    path: Path
    style: VisualStyle
    target: str
    prediction: str
    correct_characters: int
    confidences: tuple[float, ...]

    @property
    def exact(self) -> bool:
        return self.target == self.prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("real_directory", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="artifacts/real_evaluation")
    parser.add_argument("--max-errors", type=int, default=40)
    return parser


def save_csv(records: list[EvaluationRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "filename",
                "style",
                "target",
                "prediction",
                "exact",
                "correct_characters",
                "confidence_1",
                "confidence_2",
                "confidence_3",
                "confidence_4",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "filename": record.filename,
                    "style": record.style,
                    "target": record.target,
                    "prediction": record.prediction,
                    "exact": record.exact,
                    "correct_characters": record.correct_characters,
                    **{
                        f"confidence_{index + 1}": f"{confidence:.6f}"
                        for index, confidence in enumerate(record.confidences)
                    },
                }
            )


def save_error_grid(records: list[EvaluationRecord], path: Path, maximum: int) -> None:
    errors = [record for record in records if not record.exact][:maximum]
    columns = min(4, max(1, len(errors)))
    rows = max(1, (len(errors) + columns - 1) // columns)
    cell_width = 216
    cell_height = 82
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    if not errors:
        draw.text((8, 8), "No errors", fill="green", font=font)

    for index, record in enumerate(errors):
        column = index % columns
        row = index // columns
        x = column * cell_width + 8
        y = row * cell_height + 4
        with Image.open(record.path) as source:
            sheet.paste(source.convert("RGB"), (x, y))
        draw.text(
            (x, y + 54),
            f"{record.target} -> {record.prediction} | {record.style}",
            fill="red",
            font=font,
        )
    sheet.save(path)


def build_summary(records: list[EvaluationRecord]) -> str:
    character_total = sum(len(record.target) for record in records)
    character_correct = sum(record.correct_characters for record in records)
    exact_correct = sum(record.exact for record in records)
    position_correct = [0] * len(records[0].target)
    style_counts: dict[VisualStyle, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    confusions: Counter[tuple[str, str]] = Counter()

    for record in records:
        for position, (expected, predicted) in enumerate(zip(record.target, record.prediction)):
            position_correct[position] += expected == predicted
            if expected != predicted:
                confusions[(expected, predicted)] += 1
        style_result = style_counts[record.style]
        style_result[0] += record.correct_characters
        style_result[1] += len(record.target)
        style_result[2] += record.exact
        style_result[3] += 1

    lines = [
        f"Images: {len(records)}",
        (
            f"Character accuracy: {character_correct}/{character_total} "
            f"= {character_correct / character_total:.2%}"
        ),
        f"Exact accuracy: {exact_correct}/{len(records)} = {exact_correct / len(records):.2%}",
        "",
        "Position accuracy:",
    ]
    lines.extend(
        f"  position {position + 1}: {correct}/{len(records)} = {correct / len(records):.2%}"
        for position, correct in enumerate(position_correct)
    )
    lines.append("")
    lines.append("Style accuracy:")
    for visual_style in VISUAL_STYLES:
        correct_chars, total_chars, correct_images, total_images = style_counts[visual_style]
        if not total_images:
            continue
        lines.append(
            f"  {visual_style}: char={correct_chars}/{total_chars}="
            f"{correct_chars / total_chars:.2%}, exact={correct_images}/{total_images}="
            f"{correct_images / total_images:.2%}"
        )
    lines.append("")
    lines.append("Top confusions:")
    lines.extend(
        f"  {expected} -> {predicted}: {count}"
        for (expected, predicted), count in confusions.most_common(15)
    )
    return "\n".join(lines)


def main() -> None:
    args = build_parser().parse_args()
    if args.max_errors <= 0:
        raise ValueError("max errors must be positive")
    device = resolve_device(args.device)
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("model_kind") != "project_cnn":
        raise ValueError("checkpoint is not a project_cnn model")

    style = ProjectCaptchaStyle(**checkpoint["style"])
    characters: str = checkpoint["characters"]
    dataset = RealCaptchaDataset(
        args.real_directory,
        characters=characters,
        length=style.length,
        expected_size=(style.width, style.height),
    )
    model = ProjectCaptchaCNN(
        n_classes=len(characters),
        label_length=style.length,
        input_size=(style.height, style.width),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    records: list[EvaluationRecord] = []
    with torch.no_grad():
        for index, (image_path, target_text) in enumerate(dataset.samples):
            image, _ = dataset[index]
            probabilities = torch.softmax(model(image.unsqueeze(0).to(device)), dim=-1)[0]
            confidences, predicted_indices = probabilities.max(dim=-1)
            prediction = decode_indices(predicted_indices.cpu().tolist(), characters)
            records.append(
                EvaluationRecord(
                    filename=image_path.name,
                    path=image_path,
                    style=classify_real_style(measure_real_image(image_path)),
                    target=target_text,
                    prediction=prediction,
                    correct_characters=sum(
                        expected == predicted
                        for expected, predicted in zip(target_text, prediction)
                    ),
                    confidences=tuple(confidences.cpu().tolist()),
                )
            )

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(records)
    summary_path = output_dir / "summary.txt"
    csv_path = output_dir / "predictions.csv"
    error_grid_path = output_dir / "errors.png"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    save_csv(records, csv_path)
    save_error_grid(records, error_grid_path, args.max_errors)

    print(summary)
    print(f"\nSummary: {summary_path}")
    print(f"Predictions: {csv_path}")
    print(f"Error grid: {error_grid_path}")


if __name__ == "__main__":
    main()
