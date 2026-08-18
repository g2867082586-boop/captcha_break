"""Lesson 1: the three most useful ImageCaptcha APIs."""

from io import BytesIO
from pathlib import Path

from captcha.image import ImageCaptcha
from PIL import Image

import numpy as np
import torch

def main() -> None:
    output_dir = Path("artifacts")
    output_dir.mkdir(exist_ok=True)
    generator = ImageCaptcha(width=192, height=64)

    # 1. Return a PIL.Image object, useful when training a model in memory.
    pil_image = generator.generate_image("GJ66")
    print(type(pil_image))
    print(pil_image.size)
    print(pil_image.mode)
    pil_image.save(output_dir / "api_generate_image.png")

    image_array = np.asarray(pil_image)

    print("数组类型：", type(image_array))
    print("数组形状：", image_array.shape)
    print("数据类型：", image_array.dtype)
    print("最小像素值：", image_array.min())
    print("最大像素值：", image_array.max())
    print("左上角像素：", image_array[0, 0])

    # PIL 图片 → NumPy 数组
    image_array = np.asarray(pil_image)

    # uint8 的 0～255 → float32 的 0～1
    normalized_array = image_array.astype(np.float32) / 255.0

    # NumPy 数组 → PyTorch 张量
    image_tensor = torch.from_numpy(normalized_array)

    # (高度, 宽度, 通道) → (通道, 高度, 宽度)
    image_tensor = image_tensor.permute(2, 0, 1).contiguous()

    print("NumPy 形状：", image_array.shape)
    print("归一化形状：", normalized_array.shape)
    print("张量形状：", image_tensor.shape)
    print("张量类型：", image_tensor.dtype)
    print("数据类型：", image_tensor.dtype)
    print("最小值：", image_tensor.min().item())
    print("最大值：", image_tensor.max().item())

    # 2. Return an in-memory byte stream, useful for HTTP responses or uploads.
    stream: BytesIO = generator.generate("B8M3")
    Image.open(stream).save(output_dir / "api_generate_stream.png")

    # 3. Generate and write directly to a file.
    generator.write("C9N4", output_dir / "api_write.png")
    print(f"Created three examples in {output_dir.resolve()}")



if __name__ == "__main__":
    main()
