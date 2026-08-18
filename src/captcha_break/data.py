"""On-demand captcha datasets for PyTorch training."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .codec import encode_text
from .config import CaptchaConfig
from .generator import generate_captcha

Task = Literal["cnn", "ctc"]


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a normalized ``C x H x W`` float tensor."""

    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class CaptchaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Generate a new random captcha whenever an item is requested."""

    def __init__(self, config: CaptchaConfig, size: int, task: Task = "ctc") -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if task not in ("cnn", "ctc"):
            raise ValueError("task must be 'cnn' or 'ctc'")
        self.config = config
        self.size = size
        self.task = task

    @property
    def characters(self) -> str:
        return self.config.alphabet if self.task == "cnn" else self.config.ctc_characters

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index  # Samples are intentionally generated on demand.
        image, text = generate_captcha(self.config)
        target = torch.tensor(encode_text(text, self.characters), dtype=torch.long)
        return image_to_tensor(image), target
