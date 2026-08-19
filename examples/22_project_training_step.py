"""Lesson 22: perform one complete optimizer step with the project CNN."""

from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from captcha_break.codec import decode_indices
from captcha_break.data import ProjectCaptchaDataset
from captcha_break.models import ProjectCaptchaCNN
from captcha_break.project_training import fixed_length_accuracy, fixed_length_cross_entropy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    return torch.device(requested)


def gradient_norm(model: torch.nn.Module) -> float:
    squared_norm = sum(
        parameter.grad.detach().square().sum().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(squared_norm)


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.learning_rate <= 0:
        raise ValueError("batch size and learning rate must be positive")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dataset = ProjectCaptchaDataset(size=args.batch_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)

    model = ProjectCaptchaCNN(n_classes=len(dataset.characters), label_length=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    labels = [decode_indices(target.tolist(), dataset.characters) for target in targets.cpu()]

    model.eval()
    with torch.no_grad():
        baseline_logits = model(images)
        baseline_loss = fixed_length_cross_entropy(baseline_logits, targets)
        baseline_character_accuracy, baseline_exact_accuracy = fixed_length_accuracy(
            baseline_logits, targets
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    classifier_before = model.classifier.weight.detach().clone()
    logits = model(images)
    training_loss = fixed_length_cross_entropy(logits, targets)

    training_loss.backward()
    total_gradient_norm = gradient_norm(model)
    classifier_gradient = model.classifier.weight.grad
    if classifier_gradient is None:
        raise RuntimeError("classifier did not receive a gradient")
    classifier_gradient_mean = classifier_gradient.abs().mean().item()
    optimizer.step()

    classifier_update = (model.classifier.weight.detach() - classifier_before).abs().mean().item()
    model.eval()
    with torch.no_grad():
        updated_logits = model(images)
        updated_loss = fixed_length_cross_entropy(updated_logits, targets)
        updated_character_accuracy, updated_exact_accuracy = fixed_length_accuracy(
            updated_logits, targets
        )

    print(f"Device: {device}")
    print(f"Labels: {labels}")
    print(f"Images: {tuple(images.shape)}")
    print(f"Targets: {tuple(targets.shape)}")
    print(f"Logits: {tuple(logits.shape)}")
    print(f"Evaluation loss before step: {baseline_loss.item():.6f}")
    print(f"Character accuracy before: {baseline_character_accuracy:.2%}")
    print(f"Exact accuracy before: {baseline_exact_accuracy:.2%}")
    print(f"Training-mode loss used for backward: {training_loss.item():.6f}")
    print(f"Total gradient norm: {total_gradient_norm:.6f}")
    print(f"Classifier gradient mean: {classifier_gradient_mean:.8f}")
    print(f"Classifier weight update mean: {classifier_update:.8f}")
    print(f"Evaluation loss after step: {updated_loss.item():.6f}")
    print(f"Character accuracy after: {updated_character_accuracy:.2%}")
    print(f"Exact accuracy after: {updated_exact_accuracy:.2%}")


if __name__ == "__main__":
    main()
