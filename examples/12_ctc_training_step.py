import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from captcha_break.codec import (
    ctc_greedy_decode,
    decode_indices,
)
from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset
from captcha_break.models import CTCRecognizer


def main() -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前设备：", device)

    config = CaptchaConfig()

    dataset = CaptchaDataset(
        config=config,
        size=8,
        task="ctc",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
    )

    images, targets = next(iter(loader))

    images = images.to(device)
    targets = targets.to(device)

    model = CTCRecognizer(
        n_classes=len(config.ctc_characters),
        input_shape=(
            3,
            config.height,
            config.width,
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    model.train()

    # 第一步：清除旧梯度
    optimizer.zero_grad()

    # 第二步：前向传播
    logits = model(images)

    # 第三步：准备长度信息
    input_lengths = torch.full(
        size=(targets.shape[0],),
        fill_value=logits.shape[0],
        dtype=torch.long,
        device="cpu",
    )

    target_lengths = torch.full(
        size=(targets.shape[0],),
        fill_value=targets.shape[1],
        dtype=torch.long,
        device="cpu",
    )

    # 第四步：计算 CTC Loss
    loss = F.ctc_loss(
        logits.log_softmax(dim=-1),
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True,
    )

    print("CTC 输出形状：", logits.shape)
    print("标签形状：", targets.shape)
    print("CTC Loss：", loss.item())

    # 第五步：反向传播
    loss.backward()

    print(
        "分类层平均梯度：",
        model.classifier.weight.grad.abs().mean().item(),
    )

    # 第六步：更新模型参数
    optimizer.step()

    print("一次 CTC 参数更新已经完成")

    # 查看当前预测
    predicted_indices = (
        logits.detach()
        .argmax(dim=-1)
        .transpose(0, 1)
        .cpu()
    )

    targets_cpu = targets.detach().cpu()

    print("\n当前预测结果：")

    for target, prediction in zip(
        targets_cpu,
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