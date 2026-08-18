import torch

from captcha_break.config import CaptchaConfig
from captcha_break.models import FixedLengthCNN


def main() -> None:
    config = CaptchaConfig()

    model = FixedLengthCNN(
        n_classes=len(config.alphabet),
        label_length=config.length,
    )

    model.eval()

    images = torch.rand(
        2,
        3,
        config.height,
        config.width,
    )

    with torch.no_grad():
        features = model.features(images)
        pooled = model.pool(features)
        flattened = torch.flatten(pooled, 1)
        dropped = model.dropout(flattened)
        classifier_output = model.classifier(dropped)

        logits = classifier_output.view(
            images.shape[0],
            model.label_length,
            model.n_classes,
        )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

    print("卷积特征：", features.shape)
    print("自适应池化：", pooled.shape)
    print("展平：", flattened.shape)
    print("Dropout：", dropped.shape)
    print("分类层：", classifier_output.shape)
    print("最终输出：", logits.shape)
    print("概率输出：", probabilities.shape)

    print(
        "第一张图片第一个位置概率总和：",
        probabilities[0, 0].sum().item(),
    )

    # 单独观察 Dropout
    test_values = torch.ones(10_000)

    model.train()
    training_dropout = model.dropout(test_values)

    model.eval()
    prediction_dropout = model.dropout(test_values)

    training_zero_ratio = (
        training_dropout == 0
    ).float().mean().item()

    prediction_zero_ratio = (
        prediction_dropout == 0
    ).float().mean().item()

    print(
        "训练模式 Dropout 置零比例：",
        f"{training_zero_ratio:.2%}",
    )

    print(
        "预测模式 Dropout 置零比例：",
        f"{prediction_zero_ratio:.2%}",
    )


if __name__ == "__main__":
    main()