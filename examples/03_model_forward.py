import torch
from torch.utils.data import DataLoader

from captcha_break.codec import decode_indices
from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset
from captcha_break.models import FixedLengthCNN


def main() -> None:
    config = CaptchaConfig(
        width=192,
        height=64,
        length=4,
    )

    dataset = CaptchaDataset(
        config=config,
        size=10,
        task="cnn",
    )

    loader = DataLoader(
        dataset,
        batch_size=3,
        num_workers=0,
    )

    images, targets = next(iter(loader))

    model = FixedLengthCNN(
        n_classes=len(config.alphabet),
        label_length=config.length,
    )

    model.eval()

    with torch.no_grad():
        logits = model(images)

    print("输入图片形状：", images.shape)
    print("正确标签形状：", targets.shape)
    print("模型输出形状：", logits.shape)

    predicted_indices = logits.argmax(dim=-1)

    print("\n预测结果：")

    for target, prediction in zip(targets, predicted_indices):
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