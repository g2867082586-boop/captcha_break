# captcha_break：Python / VS Code 学习版

这是从 [ypwhs/captcha_break](https://github.com/ypwhs/captcha_break) 迁移出的现代 Python 工程。原仓库的核心代码放在 Jupyter Notebook 中，并依赖 TensorFlow 1.x 或硬编码 CUDA；本版本把可执行代码拆成普通 `.py` 文件，使用当前 PyTorch API，同时提供可直接运行和调试的 VS Code 配置。

原 Notebook 和预训练的旧 Keras 模型仍保留在仓库根目录，原说明见 [README_original.md](README_original.md)。它们适合对照算法历史，不再作为推荐运行入口。

## 已完成的迁移

- 使用 `src/` 布局组织 Python 包，训练代码不再依赖 Notebook 单元格的执行顺序。
- 保留两条识别路线：定长多分类 CNN，以及 CNN + 双向 LSTM + CTC。
- 去除 TensorFlow 1.x、`%matplotlib inline`、`fit_generator` 和强制 `.cuda()`。
- 自动选择 CPU 或 CUDA；Windows 默认单进程加载数据，避免 VS Code 调试时递归启动进程。
- 模型使用 `state_dict` 检查点，包含模型类型、图像配置和验证准确率。
- 提供命令行、普通 Python 脚本、测试、Ruff 检查以及 VS Code 调试/任务配置。

## 在 VS Code 中开始

本目录已经创建了 `.venv`。用 VS Code 打开 `D:\Project2\captcha_break` 后，Python 扩展会自动选择：

```text
.venv\Scripts\python.exe
```

首次在另一台机器安装时，在 VS Code 终端运行：

```powershell
uv sync --extra dev
```

生成一张验证码：

```powershell
uv run python -m captcha_break sample --text A7K2 --output artifacts/sample.png
```

也可以打开“运行和调试”，选择“01 - 生成验证码”后按 `F5`。

## 安装和验证训练功能

PyTorch 体积较大，因此训练依赖单独分组：

```powershell
uv sync --extra train --extra dev
uv run pytest
```

用很小的数据量检查完整训练流程：

```powershell
uv run python -m captcha_break train --model cnn --epochs 1 --batch-size 8 --steps-per-epoch 5 --validation-steps 2
uv run python -m captcha_break train --model ctc --epochs 1 --batch-size 8 --steps-per-epoch 5 --validation-steps 2
```

正式训练时增大 `--epochs`、`--batch-size` 和 `--steps-per-epoch`。程序会用 `torch.cuda.is_available()` 自动判断是否能使用显卡；也可显式传入 `--device cpu` 或 `--device cuda`。

训练完成后预测：

```powershell
uv run python -m captcha_break predict artifacts/ctc_best.pt
uv run python -m captcha_break predict artifacts/ctc_best.pt --image path/to/captcha.png
```

## 普通 Python 脚本入口

不想使用 `python -m` 时，可以直接运行：

```powershell
uv run python scripts/generate_sample.py
uv run python scripts/train_cnn.py --epochs 1
uv run python scripts/train_ctc.py --epochs 1
```

## 代码地图

```text
src/captcha_break/
├── config.py       # 字符集、图片尺寸、验证码长度
├── generator.py    # captcha.ImageCaptcha 的轻量封装
├── codec.py        # 字符与类别编号、CTC 贪心解码
├── data.py         # 按需生成训练图片的 PyTorch Dataset
├── models.py       # 定长 CNN 与 CNN-RNN-CTC
├── training.py     # 训练、验证、保存、载入和预测
└── cli.py          # 命令行入口

examples/
└── 01_captcha_image.py  # captcha 库三个核心图片 API
```

建议按 `generator.py → codec.py → data.py → models.py → training.py` 的顺序阅读。

## 关于旧模型

`ctc_2017.h5` 是旧版 Keras/TensorFlow 保存的模型，不能直接载入当前 PyTorch 网络。它被保留用于溯源；新训练结果保存为 `artifacts/*.pt`。

## 学习路线

我们会先学习 `captcha` 库本身，再进入识别模型：

1. `ImageCaptcha`、`generate_image()`、`generate()`、`write()`。
2. 自定义字符集、宽高、字体和噪声，理解 PIL 图片对象。
3. 将图片转为 NumPy 数组和 PyTorch 张量。
4. 理解 `Dataset` 如何无限生成随机训练样本。
5. 学定长 CNN 的多位置分类。
6. 学 CTC 的 blank、重复折叠和序列损失。

第一课的可运行材料是 [examples/01_captcha_image.py](examples/01_captcha_image.py)，详细提纲见 [docs/LEARNING_GUIDE.md](docs/LEARNING_GUIDE.md)。
