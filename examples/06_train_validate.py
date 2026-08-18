from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset
from captcha_break.models import FixedLengthCNN



def calculate_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[int, int, int, int]:
    predictions = logits.argmax(dim=-1)

    character_correct = (predictions == targets).sum().item()
    character_count = targets.numel()

    captcha_correct = (
        predictions == targets
    ).all(dim=1).sum().item()

    captcha_count = targets.shape[0]

    return (
        character_correct,
        character_count,
        captcha_correct,
        captcha_count,
    )


def train_one_epoch(
    model: FixedLengthCNN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float, float]:
    # 启用训练模式：Dropout 和 BatchNorm 按训练方式工作
    model.train()

    total_loss = 0.0
    character_correct = 0
    character_count = 0
    captcha_correct = 0
    captcha_count = 0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(images)

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
        )

        loss.backward()
        optimizer.step()

        batch_metrics = calculate_accuracy(logits, targets)

        character_correct += batch_metrics[0]
        character_count += batch_metrics[1]
        captcha_correct += batch_metrics[2]
        captcha_count += batch_metrics[3]

        total_loss += loss.item() * targets.shape[0]

    average_loss = total_loss / captcha_count
    character_accuracy = character_correct / character_count
    captcha_accuracy = captcha_correct / captcha_count

    return average_loss, character_accuracy, captcha_accuracy


def validate(
    model: FixedLengthCNN,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float, float]:
    # 启用评估模式：关闭 Dropout，固定 BatchNorm
    model.eval()

    total_loss = 0.0
    character_correct = 0
    character_count = 0
    captcha_correct = 0
    captcha_count = 0

    # 验证阶段不计算梯度
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)

            loss = F.cross_entropy(
                logits.transpose(1, 2),
                targets,
            )

            batch_metrics = calculate_accuracy(logits, targets)

            character_correct += batch_metrics[0]
            character_count += batch_metrics[1]
            captcha_correct += batch_metrics[2]
            captcha_count += batch_metrics[3]

            total_loss += loss.item() * targets.shape[0]

    average_loss = total_loss / captcha_count
    character_accuracy = character_correct / character_count
    captcha_accuracy = captcha_correct / captcha_count

    return average_loss, character_accuracy, captcha_accuracy


def main() -> None:
    config = CaptchaConfig()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前训练设备：", device)

    train_dataset = CaptchaDataset(
        config=config,
        size=128,
        task="cnn",
    )

    valid_dataset = CaptchaDataset(
        config=config,
        size=32,
        task="cnn",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=16,
        num_workers=0,
    )

    model = FixedLengthCNN(
        n_classes=len(config.alphabet),
        label_length=config.length,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1,
    )

    epochs = 8

    checkpoint_path = Path("artifacts/cnn_best.pt")
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_valid_loss = float("inf")

    patience = 4
    min_delta = 0.0001
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"\nEpoch {epoch} 开始，"
            f"当前学习率：{current_lr}"
        )

        train_loss, train_char_acc, train_captcha_acc = (
            train_one_epoch(model, train_loader, optimizer, device)
        )

        valid_loss, valid_char_acc, valid_captcha_acc = (
            validate(model, valid_loader, device)
        )

        print(f"Epoch {epoch}/{epochs}")
        print(
            f"训练：loss={train_loss:.4f}，"
            f"字符准确率={train_char_acc:.2%}，"
            f"整图准确率={train_captcha_acc:.2%}"
        )
        print(
            f"验证：loss={valid_loss:.4f}，"
            f"字符准确率={valid_char_acc:.2%}，"
            f"整图准确率={valid_captcha_acc:.2%}"
        )

        improved = (
            valid_loss < best_valid_loss - min_delta
        )

        if improved:
            best_valid_loss = valid_loss
            epochs_without_improvement = 0

            checkpoint = {
                "model_kind": "cnn",
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": asdict(config),
                "valid_loss": valid_loss,
                "valid_character_accuracy": valid_char_acc,
                "valid_captcha_accuracy": valid_captcha_acc,
            }

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print(
                f"验证损失下降，最佳模型已保存到："
                f"{checkpoint_path.resolve()}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "验证损失没有改善："
                f"{epochs_without_improvement}/{patience}"
            )

        scheduler.step()

        print("-" * 60)

        if epochs_without_improvement >= patience:
            print(
                f"验证损失连续 {patience} 个 Epoch "
                "没有改善，提前结束训练。"
            )
            break


if __name__ == "__main__":
    main()