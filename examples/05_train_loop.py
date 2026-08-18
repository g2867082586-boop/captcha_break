import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset
from captcha_break.models import FixedLengthCNN


def main() -> None:
    config = CaptchaConfig()

    dataset = CaptchaDataset(
        config=config,
        size=64,
        task="cnn",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
    )

    model = FixedLengthCNN(
        n_classes=len(config.alphabet),
        label_length=config.length,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    epochs = 2

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        captcha_correct = 0
        character_correct = 0
        sample_count = 0
        character_count = 0

        for batch_index, (images, targets) in enumerate(loader, start=1):
            optimizer.zero_grad()

            logits = model(images)

            loss = F.cross_entropy(
                logits.transpose(1, 2),
                targets,
            )

            loss.backward()
            optimizer.step()

            predictions = logits.argmax(dim=-1)

            batch_size = targets.shape[0]

            # 整张验证码正确：4个字符必须全部预测正确
            captcha_correct += (
                predictions == targets
            ).all(dim=1).sum().item()

            # 单个字符正确
            character_correct += (
                predictions == targets
            ).sum().item()

            sample_count += batch_size
            character_count += targets.numel()
            total_loss += loss.item() * batch_size

            print(
                f"Epoch {epoch} "
                f"Batch {batch_index}/{len(loader)} "
                f"Loss {loss.item():.4f}"
            )

        average_loss = total_loss / sample_count
        captcha_accuracy = captcha_correct / sample_count
        character_accuracy = character_correct / character_count

        print()
        print(f"Epoch {epoch} 完成")
        print(f"平均损失：{average_loss:.4f}")
        print(f"字符准确率：{character_accuracy:.2%}")
        print(f"整图准确率：{captcha_accuracy:.2%}")
        print("-" * 40)


if __name__ == "__main__":
    main()