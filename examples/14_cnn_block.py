import torch
from torch import nn

from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset


def print_tensor_info(
    name: str,
    tensor: torch.Tensor,
) -> None:
    print(
        f"{name:12s}",
        f"形状={tuple(tensor.shape)}",
        f"均值={tensor.mean().item():.4f}",
        f"标准差={tensor.std().item():.4f}",
        f"最小值={tensor.min().item():.4f}",
        f"最大值={tensor.max().item():.4f}",
    )


def main() -> None:
    config = CaptchaConfig()

    dataset = CaptchaDataset(
        config=config,
        size=1,
        task="cnn",
    )

    image, _ = dataset[0]
    images = image.unsqueeze(0)

    convolution = nn.Conv2d(
        in_channels=3,
        out_channels=32,
        kernel_size=3,
        padding=1,
    )

    batch_norm = nn.BatchNorm2d(32)
    activation = nn.ReLU()
    pooling = nn.MaxPool2d(2)

    with torch.no_grad():
        convolution_output = convolution(images)
        normalized_output = batch_norm(
            convolution_output
        )
        activated_output = activation(
            normalized_output
        )
        pooled_output = pooling(
            activated_output
        )

    print_tensor_info("输入", images)
    print_tensor_info(
        "卷积后",
        convolution_output,
    )
    print_tensor_info(
        "BatchNorm后",
        normalized_output,
    )
    print_tensor_info(
        "ReLU后",
        activated_output,
    )
    print_tensor_info(
        "MaxPool后",
        pooled_output,
    )


if __name__ == "__main__":
    main()