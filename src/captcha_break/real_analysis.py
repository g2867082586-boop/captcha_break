"""Pixel measurements and approximate style classification for real captchas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .project_generator import VisualStyle


@dataclass(frozen=True, slots=True)
class RealImageMetrics:
    mean_gray: float
    dark_ratio: float
    foreground_ratio: float
    bbox: tuple[int, int, int, int]


def measure_image(image: Image.Image, *, name: str = "image") -> RealImageMetrics:
    """Measure foreground occupancy and gray levels in one loaded image."""

    pixels = np.asarray(image.convert("L")).copy()
    foreground_y, foreground_x = np.where(pixels < 200)
    if not len(foreground_x):
        raise ValueError(f"image has no foreground pixels: {name}")
    return RealImageMetrics(
        mean_gray=float(pixels.mean()),
        dark_ratio=float((pixels < 128).mean()),
        foreground_ratio=float((pixels < 200).mean()),
        bbox=(
            int(foreground_x.min()),
            int(foreground_y.min()),
            int(foreground_x.max() + 1),
            int(foreground_y.max() + 1),
        ),
    )


def measure_real_image(path: str | Path) -> RealImageMetrics:
    """Open and measure one real captcha image."""

    image_path = Path(path)
    with Image.open(image_path) as source:
        return measure_image(source, name=image_path.name)


def classify_real_style(metrics: RealImageMetrics) -> VisualStyle:
    """Approximate one of the three profiles using current real-data thresholds."""

    if metrics.dark_ratio >= 0.30:
        return "solid"
    gray_texture_ratio = metrics.foreground_ratio - metrics.dark_ratio
    if metrics.dark_ratio < 0.15 and gray_texture_ratio >= 0.12:
        return "noisy_outline"
    return "clean_outline"
