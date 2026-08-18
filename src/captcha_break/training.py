"""Training, evaluation, checkpoint, and prediction utilities."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .codec import ctc_greedy_decode, decode_indices
from .config import CaptchaConfig
from .data import CaptchaDataset, image_to_tensor
from .generator import generate_captcha
from .models import CTCRecognizer, FixedLengthCNN

ModelKind = Literal["cnn", "ctc"]


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device")
    return device


def build_model(kind: ModelKind, config: CaptchaConfig) -> nn.Module:
    if kind == "cnn":
        return FixedLengthCNN(len(config.alphabet), config.length)
    if kind == "ctc":
        return CTCRecognizer(
            len(config.ctc_characters),
            input_shape=(3, config.height, config.width),
        )
    raise ValueError(f"unsupported model kind: {kind}")


def _decode_batch(logits: torch.Tensor, kind: ModelKind, config: CaptchaConfig) -> list[str]:
    if kind == "cnn":
        predictions = logits.argmax(dim=-1).detach().cpu().tolist()
        return [decode_indices(row, config.alphabet) for row in predictions]

    predictions = logits.argmax(dim=-1).transpose(0, 1).detach().cpu().tolist()
    return [ctc_greedy_decode(row, config.ctc_characters) for row in predictions]


def _targets_to_text(targets: torch.Tensor, kind: ModelKind, config: CaptchaConfig) -> list[str]:
    characters = config.alphabet if kind == "cnn" else config.ctc_characters
    return [decode_indices(row, characters) for row in targets.detach().cpu().tolist()]


def _loss(logits: torch.Tensor, targets: torch.Tensor, kind: ModelKind) -> torch.Tensor:
    if kind == "cnn":
        return F.cross_entropy(logits.transpose(1, 2), targets)

    input_lengths = torch.full((targets.shape[0],), logits.shape[0], dtype=torch.long, device="cpu")
    target_lengths = torch.full(
        (targets.shape[0],), targets.shape[1], dtype=torch.long, device="cpu"
    )
    return F.ctc_loss(
        logits.log_softmax(dim=-1),
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    kind: ModelKind,
    config: CaptchaConfig,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    progress = tqdm(loader, unit="batch", leave=False)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = _loss(logits, targets, kind)
            if training:
                loss.backward()
                optimizer.step()

            expected = _targets_to_text(targets, kind, config)
            predicted = _decode_batch(logits, kind, config)
            correct = sum(left == right for left, right in zip(expected, predicted))
            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            total_correct += correct
            total_samples += batch_size
            progress.set_postfix(
                loss=f"{total_loss / total_samples:.4f}",
                accuracy=f"{total_correct / total_samples:.2%}",
            )

    return total_loss / total_samples, total_correct / total_samples


def train_model(
    *,
    kind: ModelKind,
    config: CaptchaConfig,
    epochs: int,
    batch_size: int,
    steps_per_epoch: int,
    validation_steps: int,
    learning_rate: float,
    workers: int,
    device_name: str,
    output: str | Path,
    seed: int = 42,
) -> Path:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(device_name)
    print(f"Using device: {device}")

    train_data = CaptchaDataset(config, batch_size * steps_per_epoch, task=kind)
    valid_data = CaptchaDataset(config, batch_size * validation_steps, task=kind)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_data, shuffle=False, **loader_options)
    valid_loader = DataLoader(valid_data, shuffle=False, **loader_options)

    model = build_model(kind, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    best_accuracy = -1.0
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, kind, config, device, optimizer)
        valid_loss, valid_accuracy = run_epoch(model, valid_loader, kind, config, device)
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train loss={train_loss:.4f}, accuracy={train_accuracy:.2%}; "
            f"valid loss={valid_loss:.4f}, accuracy={valid_accuracy:.2%}"
        )
        if valid_accuracy >= best_accuracy:
            best_accuracy = valid_accuracy
            torch.save(
                {
                    "format_version": 1,
                    "model_kind": kind,
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "validation_accuracy": valid_accuracy,
                },
                output_path,
            )

    print(f"Best checkpoint: {output_path}")
    return output_path


def predict(
    checkpoint_path: str | Path,
    *,
    image_path: str | Path | None = None,
    device_name: str = "auto",
) -> tuple[str | None, str, Image.Image]:
    device = resolve_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    kind: ModelKind = checkpoint["model_kind"]
    config = CaptchaConfig(**checkpoint["config"])
    model = build_model(kind, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if image_path is None:
        image, expected = generate_captcha(config)
    else:
        image = Image.open(image_path).convert("RGB").resize((config.width, config.height))
        expected = None

    tensor = image_to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = _decode_batch(model(tensor), kind, config)[0]
    return expected, predicted, image
