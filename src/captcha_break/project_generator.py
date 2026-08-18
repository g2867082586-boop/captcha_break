"""Synthetic generator tailored to the project's simple, overlapping captchas."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from .config import DEFAULT_ALPHABET


@dataclass(frozen=True, slots=True)
class ProjectCaptchaStyle:
    """Visual controls for the 200 x 50 project captcha generator."""

    width: int = 200
    height: int = 50
    length: int = 4
    alphabet: str = DEFAULT_ALPHABET
    font_path: str | None = None
    font_size: int = 60
    horizontal_scale: float = 1.35
    rotation_degrees: float = 7.0
    vertical_jitter: int = 3
    overlap_min: int = 5
    overlap_max: int = 12
    stroke_width: int = 1
    fill_gray_min: int = 245
    fill_gray_max: int = 255
    stroke_gray_min: int = 0
    stroke_gray_max: int = 25
    background_gray: int = 255
    background_noise_density: float = 0.001
    foreground_noise_density: float = 0.0
    scratch_count: int = 0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet cannot contain duplicate characters")
        if self.font_size <= 0 or self.horizontal_scale <= 0:
            raise ValueError("font size and horizontal scale must be positive")
        if self.rotation_degrees < 0 or self.vertical_jitter < 0:
            raise ValueError("rotation and jitter cannot be negative")
        if not 0 <= self.overlap_min <= self.overlap_max:
            raise ValueError("overlap range is invalid")
        if not 0 <= self.background_gray <= 255:
            raise ValueError("background gray must be between 0 and 255")
        if not 0 <= self.fill_gray_min <= self.fill_gray_max <= 255:
            raise ValueError("fill gray range is invalid")
        if not 0 <= self.stroke_gray_min <= self.stroke_gray_max <= 255:
            raise ValueError("stroke gray range is invalid")
        if self.stroke_width < 0 or self.scratch_count < 0:
            raise ValueError("stroke width and scratch count cannot be negative")
        for density in (self.background_noise_density, self.foreground_noise_density):
            if not 0 <= density <= 1:
                raise ValueError("noise density must be between 0 and 1")


def _resolve_font_path(explicit_path: str | None) -> str:
    candidates = [
        explicit_path,
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSerif.ttf",
        "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.is_file():
            return str(path.resolve())
        try:
            ImageFont.truetype(candidate, 12)
        except OSError:
            continue
        return candidate
    raise FileNotFoundError("No usable TrueType font was found")


class ProjectCaptchaGenerator:
    """Generate labels and grayscale images that resemble the supplied samples."""

    def __init__(self, style: ProjectCaptchaStyle | None = None) -> None:
        self.style = style or ProjectCaptchaStyle()
        self.font_path = _resolve_font_path(self.style.font_path)
        self.font = ImageFont.truetype(self.font_path, self.style.font_size)

    def _render_character(self, character: str, rng: random.Random) -> Image.Image:
        stroke = self.style.stroke_width
        bbox = self.font.getbbox(character, stroke_width=stroke)
        padding = max(4, stroke + 3)
        width = bbox[2] - bbox[0] + padding * 2
        height = bbox[3] - bbox[1] + padding * 2
        glyph = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(glyph)

        fill_gray = rng.randint(self.style.fill_gray_min, self.style.fill_gray_max)
        stroke_gray = rng.randint(self.style.stroke_gray_min, self.style.stroke_gray_max)
        draw.text(
            (padding - bbox[0], padding - bbox[1]),
            character,
            font=self.font,
            fill=(fill_gray, fill_gray, fill_gray, 255),
            stroke_width=stroke,
            stroke_fill=(stroke_gray, stroke_gray, stroke_gray, 255),
        )

        scaled_width = max(1, round(glyph.width * self.style.horizontal_scale))
        glyph = glyph.resize((scaled_width, glyph.height), Image.Resampling.BICUBIC)
        angle = rng.uniform(-self.style.rotation_degrees, self.style.rotation_degrees)
        glyph = glyph.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255, 0),
        )
        # Rotation adds transparent margins.  Remove them so that ``overlap``
        # measures the distance between visible glyphs instead of empty pixels.
        visible_bbox = glyph.getchannel("A").getbbox()
        return glyph.crop(visible_bbox) if visible_bbox else glyph

    def _place_characters(
        self,
        canvas: Image.Image,
        coverage: Image.Image,
        label: str,
        rng: random.Random,
    ) -> None:
        glyphs = [self._render_character(character, rng) for character in label]
        overlaps = [
            rng.randint(self.style.overlap_min, self.style.overlap_max)
            for _ in range(len(glyphs) - 1)
        ]

        total_width = sum(glyph.width for glyph in glyphs) - sum(overlaps)
        available_width = self.style.width
        if total_width > available_width:
            scale = available_width / total_width
            glyphs = [
                glyph.resize(
                    (max(1, round(glyph.width * scale)), glyph.height),
                    Image.Resampling.BICUBIC,
                )
                for glyph in glyphs
            ]
            overlaps = [max(1, round(overlap * scale)) for overlap in overlaps]
            total_width = sum(glyph.width for glyph in glyphs) - sum(overlaps)

        x = max(0, (self.style.width - total_width) // 2)
        for index, glyph in enumerate(glyphs):
            y = (self.style.height - glyph.height) // 2
            y += rng.randint(-self.style.vertical_jitter, self.style.vertical_jitter)
            alpha = glyph.getchannel("A")
            canvas.paste(glyph.convert("L"), (x, y), alpha)

            layer_mask = Image.new("L", canvas.size, 0)
            layer_mask.paste(alpha, (x, y))
            coverage.paste(ImageChops.lighter(coverage, layer_mask))

            if index < len(overlaps):
                x += glyph.width - overlaps[index]

    def _add_noise(
        self,
        canvas: Image.Image,
        coverage: Image.Image,
        rng: random.Random,
    ) -> None:
        draw = ImageDraw.Draw(canvas)
        mask = np.asarray(coverage)
        background_points = np.argwhere(mask < 16)
        foreground_points = np.argwhere(mask > 64)

        background_count = round(
            self.style.width * self.style.height * self.style.background_noise_density
        )
        for _ in range(background_count):
            if not len(background_points):
                break
            y, x = background_points[rng.randrange(len(background_points))]
            gray = rng.randint(145, 232)
            radius = 1 if rng.random() < 0.12 else 0
            draw.ellipse(
                (int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius),
                fill=gray,
            )

        foreground_count = round(len(foreground_points) * self.style.foreground_noise_density)
        for _ in range(foreground_count):
            if not len(foreground_points):
                break
            y, x = foreground_points[rng.randrange(len(foreground_points))]
            canvas.putpixel((int(x), int(y)), rng.randint(20, 175))

        for _ in range(self.style.scratch_count):
            x = rng.randrange(self.style.width)
            y = rng.randrange(self.style.height)
            length = rng.randint(1, 4)
            draw.line(
                (x, y, min(self.style.width - 1, x + length), y + rng.choice((-1, 0, 1))),
                fill=rng.randint(170, 232),
                width=1,
            )

    def generate(
        self,
        text: str | None = None,
        *,
        rng: random.Random | None = None,
    ) -> tuple[Image.Image, str]:
        """Return one grayscale captcha and its four-character label."""

        source = rng or random.Random()
        label = (
            text
            if text is not None
            else "".join(source.choice(self.style.alphabet) for _ in range(self.style.length))
        )
        if len(label) != self.style.length:
            raise ValueError(f"text must contain exactly {self.style.length} characters")
        invalid = sorted(set(label) - set(self.style.alphabet))
        if invalid:
            raise ValueError(f"text contains characters outside the alphabet: {invalid}")

        canvas = Image.new(
            "L",
            (self.style.width, self.style.height),
            self.style.background_gray,
        )
        coverage = Image.new("L", canvas.size, 0)
        self._place_characters(canvas, coverage, label, source)
        self._add_noise(canvas, coverage, source)
        return canvas, label
