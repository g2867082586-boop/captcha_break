"""Lesson 32: fine-tune a project CNN on augmented real captchas."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from captcha_break.data import AugmentedRealCaptchaDataset, RealCaptchaDataset
from captcha_break.models import ProjectCaptchaCNN
from captcha_break.project_generator import ProjectCaptchaStyle
from captcha_break.project_training import FixedLengthMetrics, run_fixed_length_epoch
from captcha_break.training import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("train_directory", type=Path)
    parser.add_argument("validation_directory", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2032)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="artifacts/project_cnn_real_finetuned.pt")
    return parser


def metric_score(metrics: FixedLengthMetrics) -> tuple[float, float, float]:
    return metrics.exact_accuracy, metrics.character_accuracy, -metrics.loss


def format_metrics(name: str, metrics: FixedLengthMetrics) -> str:
    return (
        f"{name}: loss={metrics.loss:.4f}, "
        f"char={metrics.character_accuracy:.2%}, exact={metrics.exact_accuracy:.2%}"
    )


def save_checkpoint(
    path: Path,
    source: dict[str, object],
    model: ProjectCaptchaCNN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation: FixedLengthMetrics,
    train_directory: Path,
    validation_directory: Path,
) -> None:
    torch.save(
        {
            **source,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "fine_tuned_from_epoch": source.get("epoch"),
            "fine_tune_epoch": epoch,
            "real_train_directory": str(train_directory),
            "real_validation_directory": str(validation_directory),
            "real_validation": asdict(validation),
        },
        path,
    )


def main() -> None:
    args = build_parser().parse_args()
    for name, value in (
        ("epochs", args.epochs),
        ("batch size", args.batch_size),
        ("repeats", args.repeats),
        ("learning rate", args.learning_rate),
        ("patience", args.patience),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    source_path = args.checkpoint.expanduser().resolve()
    source = torch.load(source_path, map_location=device, weights_only=True)
    if source.get("model_kind") != "project_cnn":
        raise ValueError("checkpoint is not a project_cnn model")

    characters: str = source["characters"]
    style = ProjectCaptchaStyle(**source["style"])
    train_directory = args.train_directory.expanduser().resolve()
    validation_directory = args.validation_directory.expanduser().resolve()
    real_train = RealCaptchaDataset(
        train_directory,
        characters=characters,
        length=style.length,
        expected_size=(style.width, style.height),
    )
    train_dataset = AugmentedRealCaptchaDataset(real_train, repeats=args.repeats)
    validation_dataset = RealCaptchaDataset(
        validation_directory,
        characters=characters,
        length=style.length,
        expected_size=(style.width, style.height),
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = ProjectCaptchaCNN(
        n_classes=len(characters),
        label_length=style.length,
        input_size=(style.height, style.width),
    ).to(device)
    model.load_state_dict(source["model_state"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline = run_fixed_length_epoch(model, validation_loader, device)
    best_score = metric_score(baseline)
    epochs_without_improvement = 0
    save_checkpoint(
        output_path,
        source,
        model,
        optimizer,
        0,
        baseline,
        train_directory,
        validation_directory,
    )
    print(f"Device: {device}")
    print(f"Source checkpoint: {source_path}")
    print(f"Real training images: {len(real_train)} x {args.repeats} augmentations")
    print(f"Real validation images: {len(validation_dataset)}")
    print(f"Baseline {format_metrics('real valid', baseline)}")
    print(f"Output: {output_path}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_fixed_length_epoch(model, train_loader, device, optimizer)
        validation_metrics = run_fixed_length_epoch(model, validation_loader, device)
        scheduler.step(validation_metrics.loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{args.epochs} | lr={learning_rate:.2e}")
        print(f"  {format_metrics('real train', train_metrics)}")
        print(f"  {format_metrics('real valid', validation_metrics)}")

        score = metric_score(validation_metrics)
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            save_checkpoint(
                output_path,
                source,
                model,
                optimizer,
                epoch,
                validation_metrics,
                train_directory,
                validation_directory,
            )
            print("  Saved new best real-fine-tuned checkpoint.")
        else:
            epochs_without_improvement += 1
            print(f"  No improvement ({epochs_without_improvement}/{args.patience}).")
            if epochs_without_improvement >= args.patience:
                print("  Early stopping.")
                break
        print()

    print(f"Best checkpoint: {output_path}")


if __name__ == "__main__":
    main()
