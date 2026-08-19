from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from captcha_break.models import ProjectCaptchaCNN
from captcha_break.project_training import (
    fixed_length_accuracy,
    fixed_length_cross_entropy,
    run_fixed_length_epoch,
)


def test_fixed_length_cross_entropy_is_finite() -> None:
    logits = torch.randn(2, 4, 36, requires_grad=True)
    targets = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])

    loss = fixed_length_cross_entropy(logits, targets)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_fixed_length_accuracy_reports_character_and_exact_accuracy() -> None:
    targets = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    logits = torch.full((2, 4, 4), -10.0)
    predictions = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 0]])
    logits.scatter_(2, predictions.unsqueeze(-1), 10.0)

    character_accuracy, exact_accuracy = fixed_length_accuracy(logits, targets)

    assert character_accuracy == pytest.approx(7 / 8)
    assert exact_accuracy == pytest.approx(1 / 2)


def test_optimizer_step_changes_project_cnn_weights() -> None:
    torch.manual_seed(7)
    model = ProjectCaptchaCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    images = torch.rand(2, 1, 50, 200)
    targets = torch.randint(0, 36, (2, 4))
    before = model.classifier.weight.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = fixed_length_cross_entropy(model(images), targets)
    loss.backward()
    optimizer.step()

    assert model.classifier.weight.grad is not None
    assert model.classifier.weight.grad.abs().sum() > 0
    assert not torch.equal(before, model.classifier.weight)


def test_run_fixed_length_epoch_reports_both_accuracies() -> None:
    model = ProjectCaptchaCNN().eval()
    batches = [
        (
            torch.rand(2, 1, 50, 200),
            torch.randint(0, 36, (2, 4)),
        )
    ]

    metrics = run_fixed_length_epoch(model, batches, torch.device("cpu"))

    assert metrics.samples == 2
    assert metrics.loss > 0
    assert 0 <= metrics.character_accuracy <= 1
    assert 0 <= metrics.exact_accuracy <= 1
