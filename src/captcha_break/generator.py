"""Small, dependency-light wrappers around the third-party ``captcha`` package."""

from __future__ import annotations

import random
from pathlib import Path

from captcha.image import ImageCaptcha
from PIL import Image

from .config import CaptchaConfig


def random_text(config: CaptchaConfig, *, rng: random.Random | None = None) -> str:
    """Create one random label using the configured alphabet."""

    chooser = rng or random
    return "".join(chooser.choice(config.alphabet) for _ in range(config.length))


def generate_captcha(
    config: CaptchaConfig | None = None,
    *,
    text: str | None = None,
    rng: random.Random | None = None,
) -> tuple[Image.Image, str]:
    """Generate a PIL image and return it together with its label."""

    settings = config or CaptchaConfig()
    label = text if text is not None else random_text(settings, rng=rng)
    if len(label) != settings.length:
        raise ValueError(f"text must contain exactly {settings.length} characters")
    invalid = sorted(set(label) - set(settings.alphabet))
    if invalid:
        raise ValueError(f"text contains characters outside the alphabet: {invalid}")

    image = ImageCaptcha(width=settings.width, height=settings.height).generate_image(label)
    return image.convert("RGB"), label


def save_captcha(image: Image.Image, output: str | Path) -> Path:
    """Save an image, creating its parent directory when necessary."""

    path = Path(output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path
