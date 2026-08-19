"""On-demand captcha datasets for PyTorch training."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

from .codec import encode_text
from .config import CaptchaConfig
from .generator import generate_captcha
from .project_generator import ProjectCaptchaGenerator, ProjectCaptchaStyle

Task = Literal["cnn", "ctc"]
IMAGE_SUFFIXES = {".bmp", ".jfif", ".jpeg", ".jpg", ".png"}


@dataclass(frozen=True, slots=True)
class TargetedLabelSampling:
    """Replace one label position with a difficult character on some samples."""

    characters: str = "JMPTUWV"
    probability: float = 0.35
    position_weights: tuple[float, ...] = (1.0, 3.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if not self.characters or len(set(self.characters)) != len(self.characters):
            raise ValueError("target characters must be non-empty and unique")
        if not 0 <= self.probability <= 1:
            raise ValueError("target probability must be between 0 and 1")
        if not self.position_weights or any(weight < 0 for weight in self.position_weights):
            raise ValueError("position weights must be non-negative")
        if sum(self.position_weights) <= 0:
            raise ValueError("position weights must have a positive total")

    def validate_for(self, alphabet: str, length: int) -> None:
        invalid = sorted(set(self.characters) - set(alphabet))
        if invalid:
            raise ValueError(f"target characters are outside the alphabet: {invalid}")
        if len(self.position_weights) != length:
            raise ValueError(f"expected {length} position weights")

    def sample(self, alphabet: str, length: int, rng: random.Random) -> str:
        """Create a mostly uniform label with at most one targeted replacement."""

        self.validate_for(alphabet, length)
        label = [rng.choice(alphabet) for _ in range(length)]
        if rng.random() < self.probability:
            position = rng.choices(range(length), weights=self.position_weights, k=1)[0]
            label[position] = rng.choice(self.characters)
        return "".join(label)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image to a normalized ``C x H x W`` float tensor."""

    array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def grayscale_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalized ``1 x H x W`` float tensor."""

    array = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).unsqueeze(0).contiguous()


def label_from_filename(path: str | Path, characters: str, length: int = 4) -> str:
    """Read a captcha label from the filename portion before the first underscore."""

    filename = Path(path)
    label = filename.stem.split("_", maxsplit=1)[0].upper()
    if len(label) != length:
        raise ValueError(
            f"filename label must contain exactly {length} characters: {filename.name}"
        )
    invalid = sorted(set(label) - set(characters))
    if invalid:
        raise ValueError(f"filename label contains unknown characters {invalid}: {filename.name}")
    return label


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


class ProjectCaptchaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Generate project-style grayscale captchas on demand."""

    def __init__(
        self,
        size: int,
        style: ProjectCaptchaStyle | None = None,
        *,
        seed: int | None = None,
        label_sampling: TargetedLabelSampling | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.style = style or ProjectCaptchaStyle()
        self.size = size
        self.seed = seed
        self.label_sampling = label_sampling
        if self.label_sampling is not None:
            self.label_sampling.validate_for(self.style.alphabet, self.style.length)
        self.generator = ProjectCaptchaGenerator(self.style)

    @property
    def characters(self) -> str:
        return self.style.alphabet

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < self.size:
            raise IndexError(index)
        # A seed makes synthetic validation repeatable. Without one, training
        # receives a newly rendered captcha every time an index is requested.
        rng = random.Random(self.seed + index) if self.seed is not None else random.Random()
        text = (
            self.label_sampling.sample(self.style.alphabet, self.style.length, rng)
            if self.label_sampling is not None
            else None
        )
        image, text = self.generator.generate(text=text, rng=rng)
        target = torch.tensor(encode_text(text, self.characters), dtype=torch.long)
        return grayscale_image_to_tensor(image), target


class RealCaptchaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load real captchas whose first four filename characters are the label."""

    def __init__(
        self,
        directory: str | Path,
        *,
        characters: str,
        length: int = 4,
        expected_size: tuple[int, int] = (200, 50),
    ) -> None:
        self.directory = Path(directory).expanduser().resolve()
        if not self.directory.is_dir():
            raise FileNotFoundError(f"real captcha directory not found: {self.directory}")
        if length <= 0:
            raise ValueError("length must be positive")
        if not characters or len(set(characters)) != len(characters):
            raise ValueError("characters must be non-empty and unique")
        self.characters = characters
        self.length = length
        self.expected_size = expected_size

        paths = sorted(
            path
            for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not paths:
            raise FileNotFoundError(f"no captcha images found in {self.directory}")
        self.samples = [
            (path, label_from_filename(path, self.characters, self.length)) for path in paths
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, text = self.samples[index]
        with Image.open(path) as source:
            image = source.convert("L")
        if image.size != self.expected_size:
            raise ValueError(
                f"expected image size {self.expected_size}, got {image.size}: {path.name}"
            )
        target = torch.tensor(encode_text(text, self.characters), dtype=torch.long)
        return grayscale_image_to_tensor(image), target


class AugmentedRealCaptchaDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Repeat labeled real captchas with conservative online image augmentation."""

    def __init__(
        self,
        dataset: RealCaptchaDataset,
        *,
        repeats: int = 16,
        seed: int | None = None,
    ) -> None:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        self.dataset = dataset
        self.repeats = repeats
        self.seed = seed
        self.characters = dataset.characters

    def __len__(self) -> int:
        return len(self.dataset) * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        source_index = index % len(self.dataset)
        path, text = self.dataset.samples[source_index]
        rng = random.Random(self.seed + index) if self.seed is not None else random.Random()
        with Image.open(path) as source:
            image = source.convert("L")
        if image.size != self.dataset.expected_size:
            raise ValueError(
                f"expected image size {self.dataset.expected_size}, got {image.size}: {path.name}"
            )

        image = image.rotate(
            rng.uniform(-2.0, 2.0),
            resample=Image.Resampling.BICUBIC,
            translate=(rng.randint(-3, 3), rng.randint(-2, 2)),
            fillcolor=255,
        )
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.90, 1.10))
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.96, 1.04))
        if rng.random() < 0.20:
            image = image.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.1, 0.4)))

        target = torch.tensor(encode_text(text, self.characters), dtype=torch.long)
        return grayscale_image_to_tensor(image), target
