import torch
from torch import nn

from captcha_break.codec import decode_indices
from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset
from captcha_break.models import FixedLengthCNN


def main() -> None:
    config = CaptchaConfig()

    dataset = CaptchaDataset(
        config=config,
        size=1,
        task="cnn",
    )

    image, target = dataset[0]

    # 增加批次维度
    images = image.unsqueeze(0)

    model = FixedLengthCNN(
        n_classes=len(config.alphabet),
        label_length=config.length,
    )

    model.eval()

    print(
        "验证码：",
        decode_indices(
            target.tolist(),
            config.alphabet,
        ),
    )

    print("输入：", images.shape)

    x = images

    with torch.no_grad():
        for name, layer in model.features.named_children():
            x = layer(x)

            if isinstance(
                layer,
                (nn.Conv2d, nn.MaxPool2d),
            ):
                print(
                    f"{name:16s}",
                    f"{layer.__class__.__name__:12s}",
                    x.shape,
                )

        pooled = model.pool(x)
        print("自适应池化：", pooled.shape)

        flattened = torch.flatten(pooled, 1)
        print("展平：", flattened.shape)

        dropped = model.dropout(flattened)

        classifier_output = model.classifier(dropped)
        print("分类层：", classifier_output.shape)

        logits = classifier_output.view(
            images.shape[0],
            model.label_length,
            model.n_classes,
        )

        print("最终输出：", logits.shape)


if __name__ == "__main__":
    main()