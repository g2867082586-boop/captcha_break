import torch
from torch.utils.data import DataLoader

from captcha_break.codec import (
    ctc_greedy_decode,
    decode_indices,
)
from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset
from captcha_break.models import CTCRecognizer


def main() -> None:
    config = CaptchaConfig(
        width=192,
        height=64,
        length=4,
    )

    dataset = CaptchaDataset(
        config=config,
        size=3,
        task="ctc",
    )

    loader = DataLoader(
        dataset,
        batch_size=3,
        num_workers=0,
    )

    images, targets = next(iter(loader))

    model = CTCRecognizer(
        n_classes=len(config.ctc_characters),
        input_shape=(
            3,
            config.height,
            config.width,
        ),
    )

    model.eval()

    with torch.no_grad():
        cnn_features = model.features(images)

        sequence = (
            cnn_features
            .permute(3, 0, 1, 2)
            .flatten(2)
        )

        logits = model(images)

    print("输入图片形状：", images.shape)
    print("CNN 特征形状：", cnn_features.shape)
    print("LSTM 输入形状：", sequence.shape)
    print("CTC 输出形状：", logits.shape)
    print("正确标签形状：", targets.shape)

    predicted_indices = (
        logits
        .argmax(dim=-1)
        .transpose(0, 1)
    )

    print("\n预测结果：")

    for target, prediction in zip(
        targets,
        predicted_indices,
    ):
        real_text = decode_indices(
            target.tolist(),
            config.ctc_characters,
        )

        predicted_text = ctc_greedy_decode(
            prediction.tolist(),
            config.ctc_characters,
        )

        print(
            f"真实：{real_text}，"
            f"预测：{predicted_text}"
        )


if __name__ == "__main__":
    main()