"""Lesson 23: deliberately overfit one fixed project-captcha batch."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from captcha_break.codec import decode_indices
from captcha_break.data import ProjectCaptchaDataset
from captcha_break.models import ProjectCaptchaCNN
from captcha_break.project_training import fixed_length_accuracy, fixed_length_cross_entropy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    return torch.device(requested)


@torch.no_grad()
def evaluate_same_batch(
    model: ProjectCaptchaCNN,
    images: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float, float, torch.Tensor]:
    model.eval()
    logits = model(images)
    loss = fixed_length_cross_entropy(logits, targets).item()
    character_accuracy, exact_accuracy = fixed_length_accuracy(logits, targets)
    return loss, character_accuracy, exact_accuracy, logits.argmax(dim=-1)


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0 or args.steps <= 0 or args.print_every <= 0:
        raise ValueError("batch size, steps, and print interval must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")

    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    dataset = ProjectCaptchaDataset(size=args.batch_size, seed=args.seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    images, targets = next(iter(loader))
    images = images.to(device)
    targets = targets.to(device)

    model = ProjectCaptchaCNN(n_classes=len(dataset.characters), label_length=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    target_texts = [decode_indices(row.tolist(), dataset.characters) for row in targets.cpu()]

    print(f"Device: {device}")
    print(f"Fixed labels: {target_texts}")
    print("\n step | train loss | eval loss | character | exact")
    print("------|------------|-----------|-----------|-------")

    final_predictions = torch.empty_like(targets)
    completed_step = 0
    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        training_loss = fixed_length_cross_entropy(logits, targets)
        training_loss.backward()
        optimizer.step()

        should_evaluate = step == 1 or step % args.print_every == 0 or step == args.steps
        if not should_evaluate:
            continue

        eval_loss, character_accuracy, exact_accuracy, final_predictions = evaluate_same_batch(
            model, images, targets
        )
        completed_step = step
        print(
            f"{step:5d} | {training_loss.item():10.6f} | {eval_loss:9.6f} | "
            f"{character_accuracy:8.2%} | {exact_accuracy:6.2%}"
        )
        if exact_accuracy == 1.0 and eval_loss < 0.02:
            print("\nEarly stop: this fixed batch has been memorized.")
            break

    predicted_texts = [
        decode_indices(row.tolist(), dataset.characters) for row in final_predictions.cpu()
    ]
    print(f"\nCompleted steps: {completed_step}")
    print("Target -> prediction")
    for target, prediction in zip(target_texts, predicted_texts):
        marker = "OK" if target == prediction else "WRONG"
        print(f"  {target} -> {prediction}  {marker}")


if __name__ == "__main__":
    main()
