"""Training helpers for the fixed four-character project CNN."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

BatchIterable = Iterable[tuple[torch.Tensor, torch.Tensor]]


@dataclass(frozen=True, slots=True)
class FixedLengthMetrics:
    """Loss and both useful accuracy definitions for one complete pass."""

    loss: float
    character_accuracy: float
    exact_accuracy: float
    samples: int


def _validate_fixed_length_shapes(logits: torch.Tensor, targets: torch.Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape B x L x C, got {tuple(logits.shape)}")
    if targets.ndim != 2:
        raise ValueError(f"targets must have shape B x L, got {tuple(targets.shape)}")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"batch and label dimensions must match: {tuple(logits.shape)} vs "
            f"{tuple(targets.shape)}"
        )
    if logits.shape[2] < 2:
        raise ValueError("logits must contain at least two character classes")
    if targets.numel() and (targets.min() < 0 or targets.max() >= logits.shape[2]):
        raise ValueError("target contains a class index outside the logits range")


def fixed_length_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross entropy over every sample and character position."""

    _validate_fixed_length_shapes(logits, targets)
    # CrossEntropyLoss expects the class dimension second:
    # B x L x C becomes B x C x L.
    return F.cross_entropy(logits.transpose(1, 2), targets)


@torch.no_grad()
def fixed_length_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    """Return character accuracy and exact four-character accuracy."""

    _validate_fixed_length_shapes(logits, targets)
    predictions = logits.argmax(dim=-1)
    matches = predictions.eq(targets)
    character_accuracy = matches.float().mean().item()
    exact_accuracy = matches.all(dim=1).float().mean().item()
    return character_accuracy, exact_accuracy


def run_fixed_length_epoch(
    model: nn.Module,
    loader: BatchIterable,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    max_gradient_norm: float | None = 5.0,
) -> FixedLengthMetrics:
    """Train or evaluate for one pass, depending on whether an optimizer is supplied."""

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_characters = 0
    correct_characters = 0
    exact_samples = 0
    total_samples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = fixed_length_cross_entropy(logits, targets)
            if training:
                loss.backward()
                if max_gradient_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
                optimizer.step()

            predictions = logits.argmax(dim=-1)
            matches = predictions.eq(targets)
            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            correct_characters += matches.sum().item()
            total_characters += targets.numel()
            exact_samples += matches.all(dim=1).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("loader produced no samples")
    return FixedLengthMetrics(
        loss=total_loss / total_samples,
        character_accuracy=correct_characters / total_characters,
        exact_accuracy=exact_samples / total_samples,
        samples=total_samples,
    )
