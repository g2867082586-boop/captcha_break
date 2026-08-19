"""Lesson 24: train on dynamic synthetic data and validate on synthetic and real data."""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from captcha_break.data import ProjectCaptchaDataset, RealCaptchaDataset, TargetedLabelSampling
from captcha_break.models import ProjectCaptchaCNN
from captcha_break.project_generator import project_style_for_source
from captcha_break.project_training import FixedLengthMetrics, run_fixed_length_epoch
from captcha_break.training import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_directory", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--validation-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--geometry-preset", choices=("classic", "enhanced"), default="enhanced"
    )
    parser.add_argument(
        "--source-preset",
        choices=("legacy", "botdetect"),
        default="legacy",
        help="legacy keeps one font; botdetect samples a matched Windows font pool",
    )
    parser.add_argument("--target-characters", default="JMPTUWV")
    parser.add_argument("--target-probability", type=float, default=0.35)
    parser.add_argument(
        "--target-position-weights",
        type=float,
        nargs=4,
        default=(1.0, 3.0, 1.0, 1.0),
        metavar=("P1", "P2", "P3", "P4"),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", default="artifacts/project_cnn_best.pt")
    return parser


def format_metrics(name: str, metrics: FixedLengthMetrics) -> str:
    return (
        f"{name}: loss={metrics.loss:.4f}, "
        f"char={metrics.character_accuracy:.2%}, exact={metrics.exact_accuracy:.2%}"
    )


def validate_arguments(args: argparse.Namespace) -> None:
    positive_values = {
        "epochs": args.epochs,
        "batch size": args.batch_size,
        "steps per epoch": args.steps_per_epoch,
        "validation steps": args.validation_steps,
        "learning rate": args.learning_rate,
        "patience": args.patience,
    }
    invalid = [name for name, value in positive_values.items() if value <= 0]
    if invalid:
        raise ValueError(f"these arguments must be positive: {', '.join(invalid)}")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    if not 0 <= args.target_probability <= 1:
        raise ValueError("target probability must be between 0 and 1")


def main() -> None:
    args = build_parser().parse_args()
    validate_arguments(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    style = project_style_for_source(args.source_preset, args.geometry_preset)
    label_sampling = TargetedLabelSampling(
        characters=args.target_characters.upper(),
        probability=args.target_probability,
        position_weights=tuple(args.target_position_weights),
    )
    label_sampling.validate_for(style.alphabet, style.length)
    train_dataset = ProjectCaptchaDataset(
        size=args.batch_size * args.steps_per_epoch,
        style=style,
        seed=None,
        label_sampling=label_sampling,
    )
    synthetic_validation_dataset = ProjectCaptchaDataset(
        size=args.batch_size * args.validation_steps,
        style=style,
        seed=args.seed,
    )
    real_validation_dataset = RealCaptchaDataset(
        args.real_directory,
        characters=style.alphabet,
        length=style.length,
        expected_size=(style.width, style.height),
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_options)
    synthetic_validation_loader = DataLoader(
        synthetic_validation_dataset,
        shuffle=False,
        **loader_options,
    )
    real_validation_loader = DataLoader(
        real_validation_dataset,
        shuffle=False,
        **loader_options,
    )

    model = ProjectCaptchaCNN(
        n_classes=len(style.alphabet),
        label_length=style.length,
        input_size=(style.height, style.width),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_score = (-1.0, -1.0, -1.0, float("-inf"))
    epochs_without_improvement = 0

    print(f"Device: {device}")
    print(f"Geometry preset: {args.geometry_preset}")
    print(f"Source preset: {args.source_preset}")
    print(f"Training samples per epoch: {len(train_dataset)} (dynamic, seed=None)")
    print(
        f"Targeted training labels: {label_sampling.characters}, "
        f"probability={label_sampling.probability:.0%}"
    )
    print(f"Synthetic validation samples: {len(synthetic_validation_dataset)} (fixed)")
    print(f"Real validation samples: {len(real_validation_dataset)}")
    print(f"Output: {output_path}\n")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_fixed_length_epoch(model, train_loader, device, optimizer)
        synthetic_metrics = run_fixed_length_epoch(model, synthetic_validation_loader, device)
        real_metrics = run_fixed_length_epoch(model, real_validation_loader, device)
        scheduler.step(synthetic_metrics.loss)
        learning_rate = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch}/{args.epochs} | lr={learning_rate:.2e}")
        print(f"  {format_metrics('train', train_metrics)}")
        print(f"  {format_metrics('synthetic valid', synthetic_metrics)}")
        print(f"  {format_metrics('real valid', real_metrics)}")

        score = (
            real_metrics.exact_accuracy,
            real_metrics.character_accuracy,
            synthetic_metrics.exact_accuracy,
            -synthetic_metrics.loss,
        )
        if score > best_score:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "format_version": 1,
                    "model_kind": "project_cnn",
                    "geometry_preset": args.geometry_preset,
                    "source_preset": args.source_preset,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "characters": style.alphabet,
                    "style": asdict(style),
                    "label_sampling": asdict(label_sampling),
                    "synthetic_validation": asdict(synthetic_metrics),
                    "real_validation": asdict(real_metrics),
                },
                output_path,
            )
            print("  Saved new best checkpoint.")
        else:
            epochs_without_improvement += 1
            print(
                f"  No real-validation improvement ({epochs_without_improvement}/{args.patience})."
            )
            if epochs_without_improvement >= args.patience:
                print("  Early stopping.")
                break
        print()

    print(f"Best checkpoint: {output_path}")


if __name__ == "__main__":
    main()
