"""Command-line interface used by the VS Code launch configurations."""

from __future__ import annotations

import argparse
import random
from collections.abc import Sequence
from pathlib import Path

from .config import DEFAULT_ALPHABET, CaptchaConfig
from .generator import generate_captcha, save_captcha


def _config_from_args(args: argparse.Namespace) -> CaptchaConfig:
    return CaptchaConfig(
        width=args.width,
        height=args.height,
        length=args.length,
        alphabet=args.alphabet,
    )


def _add_image_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--alphabet", default=DEFAULT_ALPHABET)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="captcha-break",
        description="Generate captcha images and train CNN/CTC recognizers.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample = subparsers.add_parser("sample", help="generate one captcha image")
    _add_image_arguments(sample)
    sample.add_argument("--text", help="fixed label instead of a random label")
    sample.add_argument("--seed", type=int, default=None)
    sample.add_argument("--output", default="artifacts/sample.png")

    train = subparsers.add_parser("train", help="train a CNN or CTC recognizer")
    _add_image_arguments(train)
    train.add_argument("--model", choices=("cnn", "ctc"), default="ctc")
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--steps-per-epoch", type=int, default=100)
    train.add_argument("--validation-steps", type=int, default=20)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--output", default=None)

    predict_parser = subparsers.add_parser("predict", help="predict an image with a checkpoint")
    predict_parser.add_argument("checkpoint")
    predict_parser.add_argument("--image", default=None)
    predict_parser.add_argument("--device", default="auto")
    predict_parser.add_argument("--output", default="artifacts/prediction.png")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "sample":
        config = _config_from_args(args)
        rng = random.Random(args.seed) if args.seed is not None else None
        image, text = generate_captcha(config, text=args.text, rng=rng)
        output = save_captcha(image, args.output)
        print(f"Label: {text}")
        print(f"Image: {output}")
        return 0

    if args.command == "train":
        try:
            from .training import train_model
        except ModuleNotFoundError as error:
            if error.name in {"torch", "tqdm"}:
                raise SystemExit(
                    "Training dependencies are missing. Run: uv sync --extra train"
                ) from error
            raise

        config = _config_from_args(args)
        output = args.output or f"artifacts/{args.model}_best.pt"
        train_model(
            kind=args.model,
            config=config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
            validation_steps=args.validation_steps,
            learning_rate=args.learning_rate,
            workers=args.workers,
            device_name=args.device,
            output=output,
            seed=args.seed,
        )
        return 0

    try:
        from .training import predict
    except ModuleNotFoundError as error:
        if error.name in {"torch", "tqdm"}:
            raise SystemExit(
                "Prediction dependencies are missing. Run: uv sync --extra train"
            ) from error
        raise

    expected, predicted, image = predict(
        args.checkpoint,
        image_path=args.image,
        device_name=args.device,
    )
    output = save_captcha(image, Path(args.output))
    if expected is not None:
        print(f"Expected: {expected}")
    print(f"Predicted: {predicted}")
    print(f"Image: {output}")
    return 0
