# captcha 与验证码识别学习路线

## 第一课：先把验证码“生出来”

目标是掌握 `captcha.image.ImageCaptcha` 最常用的三个输出方式。

```python
from captcha.image import ImageCaptcha

generator = ImageCaptcha(width=192, height=64)
image = generator.generate_image("A7K2")
image.save("sample.png")
```

这里的 `image` 是一个 `PIL.Image.Image` 对象。它可以显示、裁剪、缩放、转成 NumPy 数组，也能直接交给后续的数据处理代码。

三个核心方法的区别：

| 方法 | 返回/行为 | 适用场景 |
|---|---|---|
| `generate_image(text)` | 返回 PIL 图片 | 模型训练、继续处理图片 |
| `generate(text)` | 返回内存字节流 | Web 响应、上传、不落盘处理 |
| `write(text, output)` | 直接写文件 | 快速批量生成数据文件 |

动手顺序：

1. 运行 `uv run python examples/01_captcha_image.py`。
2. 把 `A7K2` 改成自己的四位字符串。
3. 修改 `width`、`height`，观察字符拥挤程度和噪声变化。
4. 在 `src/captcha_break/generator.py` 的 `generate_captcha()` 处打断点，按 `F5` 单步查看 `label` 和 `image`。

## 第二课：字符集与标签

阅读 `config.py` 和 `codec.py`，理解：

- `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ` 为什么是 36 个类别；
- CNN 为什么直接用 0 到 35；
- CTC 为什么要在索引 0 额外放一个 `-` 作为 blank；
- `A7K2` 如何编码成整数，又如何还原。

## 第三课：从 PIL 到张量

阅读 `data.py` 的 `image_to_tensor()`：

- PIL 图片布局是 `H × W × C`；
- PyTorch 默认使用 `C × H × W`；
- 像素从 `0..255` 归一化到 `0..1`；
- `CaptchaDataset` 为什么无需提前保存几万张图片。

## 第四课：定长 CNN

阅读 `FixedLengthCNN`。它适合字符数固定、字符顺序清楚的验证码。网络共享卷积特征，最后一次输出四个位置各自的 36 类分数。

## 第五课：CTC

阅读 `CTCRecognizer` 和 `ctc_greedy_decode()`。重点理解时间轴、blank 和相邻重复字符折叠。CTC 更适合字符位置不固定或长度可变的序列识别。

## 推荐互动方式

每次学习一个文件或一个函数。你可以直接说“开始第一课”或贴出某一行代码问我；我会先解释输入和输出，再带你在 VS Code 中打断点运行，最后给一个很小的练习。

