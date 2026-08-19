"""Lesson 27: compare two project CNN checkpoints on the same real captchas."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from captcha_break.codec import decode_indices
from captcha_break.data import RealCaptchaDataset
from captcha_break.models import ProjectCaptchaCNN
from captcha_break.project_generator import ProjectCaptchaStyle
from captcha_break.real_analysis import classify_real_style, measure_real_image
from captcha_break.training import resolve_device


@dataclass(frozen=True, slots=True)
class Prediction:
    text: str
    confidences: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ComparisonRecord:
    path: Path
    target: str
    style: str
    first: Prediction
    second: Prediction

    @property
    def first_correct_characters(self) -> int:
        return sum(expected == predicted for expected, predicted in zip(self.target, self.first.text))

    @property
    def second_correct_characters(self) -> int:
        return sum(
            expected == predicted for expected, predicted in zip(self.target, self.second.text)
        )

    @property
    def character_delta(self) -> int:
        return self.second_correct_characters - self.first_correct_characters

    @property
    def category(self) -> str:
        first_exact = self.first.text == self.target
        second_exact = self.second.text == self.target
        if first_exact and second_exact:
            return "both_correct"
        if not first_exact and second_exact:
            return "second_fixed"
        if first_exact and not second_exact:
            return "second_regressed"
        return "both_wrong"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first_checkpoint", type=Path, help="baseline checkpoint, such as v1")
    parser.add_argument("second_checkpoint", type=Path, help="new checkpoint, such as v2")
    parser.add_argument("real_directory", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/checkpoint_comparison"))
    return parser


def load_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[ProjectCaptchaCNN, ProjectCaptchaStyle, str, int]:
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location=device, weights_only=True
    )
    if checkpoint.get("model_kind") != "project_cnn":
        raise ValueError(f"not a project_cnn checkpoint: {checkpoint_path}")

    style = ProjectCaptchaStyle(**checkpoint["style"])
    characters: str = checkpoint["characters"]
    model = ProjectCaptchaCNN(
        n_classes=len(characters),
        label_length=style.length,
        input_size=(style.height, style.width),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, style, characters, int(checkpoint["epoch"])


def predict(
    model: ProjectCaptchaCNN,
    image: torch.Tensor,
    characters: str,
    device: torch.device,
) -> Prediction:
    probabilities = torch.softmax(model(image.unsqueeze(0).to(device)), dim=-1)[0]
    confidences, indices = probabilities.max(dim=-1)
    return Prediction(
        text=decode_indices(indices.cpu().tolist(), characters),
        confidences=tuple(confidences.cpu().tolist()),
    )


def save_csv(records: list[ComparisonRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "filename",
                "style",
                "target",
                "first_prediction",
                "second_prediction",
                "first_correct_characters",
                "second_correct_characters",
                "character_delta",
                "category",
                "first_mean_confidence",
                "second_mean_confidence",
                "first_confidence_1",
                "first_confidence_2",
                "first_confidence_3",
                "first_confidence_4",
                "second_confidence_1",
                "second_confidence_2",
                "second_confidence_3",
                "second_confidence_4",
            ),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "filename": record.path.name,
                    "style": record.style,
                    "target": record.target,
                    "first_prediction": record.first.text,
                    "second_prediction": record.second.text,
                    "first_correct_characters": record.first_correct_characters,
                    "second_correct_characters": record.second_correct_characters,
                    "character_delta": record.character_delta,
                    "category": record.category,
                    "first_mean_confidence": f"{sum(record.first.confidences) / 4:.6f}",
                    "second_mean_confidence": f"{sum(record.second.confidences) / 4:.6f}",
                    **{
                        f"first_confidence_{position}": f"{confidence:.6f}"
                        for position, confidence in enumerate(record.first.confidences, start=1)
                    },
                    **{
                        f"second_confidence_{position}": f"{confidence:.6f}"
                        for position, confidence in enumerate(record.second.confidences, start=1)
                    },
                }
            )


def build_summary(
    records: list[ComparisonRecord], first_name: str, second_name: str, first_epoch: int, second_epoch: int
) -> str:
    image_count = len(records)
    character_count = image_count * len(records[0].target)
    first_characters = sum(record.first_correct_characters for record in records)
    second_characters = sum(record.second_correct_characters for record in records)
    first_exact = sum(record.first.text == record.target for record in records)
    second_exact = sum(record.second.text == record.target for record in records)
    categories = {
        name: sum(record.category == name for record in records)
        for name in ("both_correct", "second_fixed", "second_regressed", "both_wrong")
    }
    improved = sum(record.character_delta > 0 for record in records)
    unchanged = sum(record.character_delta == 0 for record in records)
    regressed = sum(record.character_delta < 0 for record in records)

    lines = [
        f"First:  {first_name} (saved epoch {first_epoch})",
        f"Second: {second_name} (saved epoch {second_epoch})",
        f"Images: {image_count}",
        "",
        (
            f"First character accuracy:  {first_characters}/{character_count} "
            f"= {first_characters / character_count:.2%}"
        ),
        (
            f"Second character accuracy: {second_characters}/{character_count} "
            f"= {second_characters / character_count:.2%}"
        ),
        f"Character change: {second_characters - first_characters:+d}",
        "",
        f"First exact accuracy:  {first_exact}/{image_count} = {first_exact / image_count:.2%}",
        f"Second exact accuracy: {second_exact}/{image_count} = {second_exact / image_count:.2%}",
        f"Exact change: {second_exact - first_exact:+d}",
        "",
        "Per-image character result:",
        f"  improved:  {improved}",
        f"  unchanged: {unchanged}",
        f"  regressed: {regressed}",
        "",
        "Exact-result transitions:",
        f"  both correct:     {categories['both_correct']}",
        f"  second fixed:     {categories['second_fixed']}",
        f"  second regressed: {categories['second_regressed']}",
        f"  both wrong:       {categories['both_wrong']}",
    ]
    return "\n".join(lines)


def save_comparison_grid(records: list[ComparisonRecord], path: Path) -> None:
    # Improvements appear first, regressions last, making the two groups easy to inspect.
    ordered = sorted(records, key=lambda record: (-record.character_delta, record.path.name))
    columns = 3
    cell_width = 224
    cell_height = 100
    rows = (len(ordered) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    colors = {
        "both_correct": "green",
        "second_fixed": "blue",
        "second_regressed": "red",
        "both_wrong": "#8a5500",
    }

    for index, record in enumerate(ordered):
        x = index % columns * cell_width + 8
        y = index // columns * cell_height + 4
        with Image.open(record.path) as source:
            sheet.paste(source.convert("RGB"), (x, y))
        draw.text(
            (x, y + 54),
            f"T:{record.target}  v1:{record.first.text}  v2:{record.second.text}",
            fill=colors[record.category],
            font=font,
        )
        draw.text(
            (x, y + 70),
            f"delta={record.character_delta:+d}  {record.category}",
            fill=colors[record.category],
            font=font,
        )
    sheet.save(path)


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    first_model, first_style, first_characters, first_epoch = load_model(
        args.first_checkpoint, device
    )
    second_model, second_style, second_characters, second_epoch = load_model(
        args.second_checkpoint, device
    )
    if (first_style.width, first_style.height, first_style.length) != (
        second_style.width,
        second_style.height,
        second_style.length,
    ):
        raise ValueError("the two checkpoints use different image or label sizes")

    dataset = RealCaptchaDataset(
        args.real_directory,
        characters=first_characters,
        length=first_style.length,
        expected_size=(first_style.width, first_style.height),
    )
    unsupported_by_second = sorted(
        set("".join(target for _, target in dataset.samples)) - set(second_characters)
    )
    if unsupported_by_second:
        raise ValueError(
            f"real labels contain characters unsupported by the second checkpoint: "
            f"{unsupported_by_second}"
        )
    records: list[ComparisonRecord] = []
    with torch.no_grad():
        for index, (image_path, target) in enumerate(dataset.samples):
            image, _ = dataset[index]
            records.append(
                ComparisonRecord(
                    path=image_path,
                    target=target,
                    style=classify_real_style(measure_real_image(image_path)),
                    first=predict(first_model, image, first_characters, device),
                    second=predict(second_model, image, second_characters, device),
                )
            )

    output_directory = args.output.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    summary = build_summary(
        records,
        args.first_checkpoint.name,
        args.second_checkpoint.name,
        first_epoch,
        second_epoch,
    )
    summary_path = output_directory / "summary.txt"
    csv_path = output_directory / "comparison.csv"
    grid_path = output_directory / "comparison_grid.png"
    summary_path.write_text(summary + "\n", encoding="utf-8")
    save_csv(records, csv_path)
    save_comparison_grid(records, grid_path)

    print(summary)
    print(f"\nSummary: {summary_path}")
    print(f"CSV: {csv_path}")
    print(f"Grid: {grid_path}")


if __name__ == "__main__":
    main()
