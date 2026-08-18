from captcha_break.codec import decode_indices
from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset
from torch.utils.data import DataLoader

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
        shuffle=False,
        num_workers=0,
    )

    images, targets = next(iter(loader))

    print("批次图片形状：", images.shape)
    print("批次标签形状：", targets.shape)

    print("这一批的验证码：")
    for target in targets:
        text = decode_indices(target.tolist(), dataset.characters)
        print(text)
    
    print("数据集长度：", len(dataset))
    print("字符集：", dataset.characters)

    image, target = dataset[0]

    print("图片张量形状：", image.shape)
    print("标签编号：", target)
    print("标签形状：", target.shape)

    text = decode_indices(target.tolist(), dataset.characters)
    print("验证码文字：", text)


if __name__ == "__main__":
    main()