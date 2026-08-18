from pathlib import Path

import torch

from captcha_break.codec import decode_indices
from captcha_break.config import CaptchaConfig
from captcha_break.data import image_to_tensor
from captcha_break.generator import (
    generate_captcha,
    save_captcha,
)
from captcha_break.models import FixedLengthCNN


def main() -> None:
    checkpoint_path = Path(
        "artifacts/cnn_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "找不到 artifacts/cnn_best.pt，"
            "请先运行快速 CNN 训练。"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if checkpoint["model_kind"] != "cnn":
        raise ValueError("该检查点不是 CNN 模型")

    config = CaptchaConfig(
        **checkpoint["config"],
    )

    model = FixedLengthCNN(
        n_classes=len(config.alphabet),
        label_length=config.length,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state"]
    )

    model.eval()

    image, real_text = generate_captcha(config)

    image_tensor = (
        image_to_tensor(image)
        .unsqueeze(0)
        .to(device)
    )

    # inference_mode 比 no_grad 更适合纯预测
    with torch.inference_mode():
        logits = model(image_tensor)

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        position_confidences, predicted_indices = (
            probabilities.max(dim=-1)
        )

    predicted_indices = (
        predicted_indices[0].cpu()
    )

    position_confidences = (
        position_confidences[0].cpu()
    )

    predicted_text = decode_indices(
        predicted_indices.tolist(),
        config.alphabet,
    )

    overall_confidence = (
        position_confidences.min().item()
    )

    threshold = 0.80
    accepted = overall_confidence >= threshold

    output_path = save_captcha(
        image,
        "artifacts/confidence_example.png",
    )

    print("真实验证码：", real_text)
    print("预测验证码：", predicted_text)

    print("\n每个位置的置信度：")

    for position, (character, confidence) in enumerate(
        zip(
            predicted_text,
            position_confidences.tolist(),
        ),
        start=1,
    ):
        print(
            f"位置 {position}："
            f"{character}，"
            f"置信度 {confidence:.2%}"
        )

    print(
        "\n整图最低置信度：",
        f"{overall_confidence:.2%}",
    )

    print(
        "是否接受预测：",
        "是" if accepted else "否",
    )

    print("图片位置：", output_path)


if __name__ == "__main__":
    main()