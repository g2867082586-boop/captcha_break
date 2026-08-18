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
        size=20,
        task="cnn",
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        num_workers=0,
    )

    images, targets = next(iter(loader))

    model = FixedLengthCNN(
        n_classes=len(config.alphabet),
        label_length=config.length,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    model.train()

    # 清除上一次计算留下的梯度
    optimizer.zero_grad()

    # 前向传播
    logits = model(images)

    # 计算预测结果与正确答案之间的损失
    loss = F.cross_entropy(
        logits.transpose(1, 2),
        targets,
    )

    print("模型输出形状：", logits.shape)
    print("转换后的形状：", logits.transpose(1, 2).shape)
    print("正确标签形状：", targets.shape)
    print("当前损失：", loss.item())

    # 反向传播，计算每个参数的梯度
    loss.backward()

    print(
        "分类层的平均梯度：",
        model.classifier.weight.grad.abs().mean().item(),
    )

    # 根据梯度更新模型参数
    optimizer.step()

    print("一次模型更新已经完成")


if __name__ == "__main__":
    main()