"""Lesson 21: follow one project captcha batch through the lightweight CNN."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from captcha_break.codec import decode_indices
from captcha_break.data import ProjectCaptchaDataset
from captcha_break.models import FixedLengthCNN, ProjectCaptchaCNN


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")

    dataset = ProjectCaptchaDataset(size=args.batch_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, targets = next(iter(loader))
    labels = [decode_indices(target.tolist(), dataset.characters) for target in targets]

    model = ProjectCaptchaCNN(
        n_classes=len(dataset.characters),
        label_length=4,
    ).eval()
    legacy_model = FixedLengthCNN(
        n_classes=len(dataset.characters),
        label_length=4,
    )

    print(f"Labels: {labels}")
    print(f"Input: {tuple(images.shape)}")
    print("\nFeature shapes")
    features = images
    with torch.no_grad():
        for name, block in model.features.named_children():
            features = block(features)
            print(f"  {name}: {tuple(features.shape)}")

        pooled = model.pool(features)
        flattened = torch.flatten(pooled, 1)
        classifier_output = model.classifier(model.dropout(flattened))
        logits = classifier_output.view(images.shape[0], model.label_length, model.n_classes)
        probabilities = torch.softmax(logits, dim=-1)

    print(f"\nAdaptive pool: {tuple(pooled.shape)}")
    print(f"Flatten: {tuple(flattened.shape)}")
    print(f"Classifier: {tuple(classifier_output.shape)}")
    print(f"Final logits: {tuple(logits.shape)}")
    print(f"Probability sum at [0, 0]: {probabilities[0, 0].sum().item():.6f}")
    print(f"\nProject CNN parameters: {parameter_count(model):,}")
    print(f"Legacy CNN parameters: {parameter_count(legacy_model):,}")


if __name__ == "__main__":
    main()
