from pathlib import Path

import torch

from captcha_break.codec import decode_indices
from captcha_break.config import CaptchaConfig
from captcha_break.data import image_to_tensor
from captcha_break.generator import generate_captcha, save_captcha
from captcha_break.models import FixedLengthCNN


def main() -> None:
    checkpoint_path = Path("artifacts/cnn_best.pt")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前预测设备：", device)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "找不到模型，请先运行 examples/06_train_validate.py"
        )

    # 从磁盘读取检查点
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # 恢复生成验证码时使用的配置
    config = CaptchaConfig(
        **checkpoint["config"],
    )

    # 重新创建相同结构的模型
    model = FixedLengthCNN(
        n_classes=len(config.alphabet),
        label_length=config.length,
    ).to(device)

    # 把训练得到的参数放入模型
    model.load_state_dict(
        checkpoint["model_state"],
    )

    # 切换到预测模式
    model.eval()

    # 生成一张新的验证码
    image, real_text = generate_captcha(config)

    # PIL 图片 → 张量，并增加批次维度
    image_tensor = (
        image_to_tensor(image)
        .unsqueeze(0)
        .to(device)
    )

    # 预测时不需要计算梯度
    with torch.no_grad():
        logits = model(image_tensor)

    predicted_indices = (
        logits.argmax(dim=-1)[0].cpu()
    )

    predicted_text = decode_indices(
        predicted_indices.tolist(),
        config.alphabet,
    )

    output_path = save_captcha(
        image,
        "artifacts/prediction_example.png",
    )

    print("模型训练轮次：", checkpoint["epoch"])
    print("模型验证损失：", checkpoint["valid_loss"])
    print("真实验证码：", real_text)
    print("模型预测值：", predicted_text)
    print("验证码图片：", output_path)


if __name__ == "__main__":
    main()