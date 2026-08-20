"""Lesson 40: rank project CNN checkpoints on the fixed real validation split."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from captcha_break.data import IMAGE_SUFFIXES, RealCaptchaDataset
from captcha_break.models import ProjectCaptchaCNN
from captcha_break.project_generator import ProjectCaptchaStyle
from captcha_break.project_training import run_fixed_length_epoch
from captcha_break.training import resolve_device


@dataclass(frozen=True, slots=True)
class CheckpointResult:
    checkpoint: str
    epoch: int
    loss: float
    character_accuracy: float
    exact_accuracy: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_directory", type=Path)
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/real499_checkpoint_selection.csv")
    )
    return parser


def evaluate_checkpoint(
    checkpoint_path: Path,
    validation_directory: Path,
    device: torch.device,
    batch_size: int,
) -> CheckpointResult:
    checkpoint = torch.load(
        checkpoint_path.expanduser().resolve(), map_location=device, weights_only=True
    )
    if checkpoint.get("model_kind") != "project_cnn":
        raise ValueError(f"not a project_cnn checkpoint: {checkpoint_path}")
    characters: str = checkpoint["characters"]
    style = ProjectCaptchaStyle(**checkpoint["style"])
    dataset = RealCaptchaDataset(
        validation_directory,
        characters=characters,
        length=style.length,
        expected_size=(style.width, style.height),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = ProjectCaptchaCNN(
        n_classes=len(characters),
        label_length=style.length,
        input_size=(style.height, style.width),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    metrics = run_fixed_length_epoch(model, loader, device)
    return CheckpointResult(
        checkpoint=str(checkpoint_path.expanduser().resolve()),
        epoch=int(checkpoint.get("epoch", -1)),
        loss=metrics.loss,
        character_accuracy=metrics.character_accuracy,
        exact_accuracy=metrics.exact_accuracy,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    validation_directory = args.validation_directory.expanduser().resolve()
    device = resolve_device(args.device)
    results = [
        evaluate_checkpoint(path, validation_directory, device, args.batch_size)
        for path in args.checkpoints
    ]
    results.sort(
        key=lambda item: (item.exact_accuracy, item.character_accuracy, -item.loss),
        reverse=True,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    image_count = sum(
        path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        for path in validation_directory.iterdir()
    )
    print(f"Device: {device}")
    print(f"Validation images: {image_count}")
    print("\nRanked checkpoints:")
    for rank, result in enumerate(results, start=1):
        print(
            f"{rank:>2}. {Path(result.checkpoint).name}: "
            f"exact={result.exact_accuracy:.2%}, "
            f"char={result.character_accuracy:.2%}, loss={result.loss:.4f}"
        )
    print(f"\nSelected for fine-tuning: {results[0].checkpoint}")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
