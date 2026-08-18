import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from captcha_break.codec import decode_indices
from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset
from captcha_break.models import FixedLengthCNN


def main() -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前设备：", device)

    # 使用较小宽度，提高 CPU 实验速度
    config = CaptchaConfig(
        width=96,
        height=64,
        length=4,
    )

    dataset = CaptchaDataset(
        config=config,
        size=8,
        task="cnn",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
    )

    # 只获取一次，后续反复使用相同图片
    images, targets = next(iter(loader))

    images = images.to(device)
    targets = targets.to(device)

    model = FixedLengthCNN(
        n_classes=len(config.alphabet),
        label_length=config.length,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    training_steps = 50

    for step in range(1, training_steps + 1):
        model.train()

        optimizer.zero_grad()

        logits = model(images)

        loss = F.cross_entropy(
            logits.transpose(1, 2),
            targets,
        )

        loss.backward()
        optimizer.step()

        if step == 1 or step % 10 == 0:
            model.eval()

            with torch.no_grad():
                eval_logits = model(images)
                predictions = eval_logits.argmax(dim=-1)

            character_accuracy = (
                predictions == targets
            ).float().mean().item()

            captcha_accuracy = (
                predictions == targets
            ).all(dim=1).float().mean().item()

            print(
                f"Step {step:3d} | "
                f"Loss {loss.item():.4f} | "
                f"字符准确率 {character_accuracy:.2%} | "
                f"整图准确率 {captcha_accuracy:.2%}"
            )

    print("\n最终预测：")

    predictions = predictions.cpu()
    targets = targets.cpu()

    for target, prediction in zip(targets, predictions):
        real_text = decode_indices(
            target.tolist(),
            config.alphabet,
        )

        predicted_text = decode_indices(
            prediction.tolist(),
            config.alphabet,
        )

        print(f"真实：{real_text}，预测：{predicted_text}")


if __name__ == "__main__":
    main()