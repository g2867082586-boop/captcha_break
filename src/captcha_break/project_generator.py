"""Synthetic generator tailored to the project's overlapping captchas."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, cast

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

VisualStyle = Literal["clean_outline", "noisy_outline", "solid"]
GeometryPreset = Literal["classic", "enhanced"]
SourcePreset = Literal["legacy", "botdetect"]
PROJECT_ALPHABET = "34689ABCDEHJKMNPRTUVWXY"
VISUAL_STYLES: tuple[VisualStyle, ...] = (
    "clean_outline",
    "noisy_outline",
    "solid",
)

# The three visual families observed in the real set closely match these
# documented BotDetect styles.  The names are descriptive metadata only; no
# captcha.com images or proprietary drawing code are bundled with this project.
BOTDETECT_STYLE_NAMES: dict[VisualStyle, str] = {
    "clean_outline": "Overlap2",
    "noisy_outline": "Rough",
    "solid": "BlackOverlap",
}

# BotDetect varies glyph shapes much more than the original single-font Python
# approximation.  These are ordinary Windows fonts with similar broad families.
# Missing fonts are skipped, and a portable Pillow font fallback is always used.
BOTDETECT_FONT_CANDIDATES: tuple[str, ...] = (
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/timesi.ttf",
    "C:/Windows/Fonts/BOOKOS.TTF",
    "C:/Windows/Fonts/BOOKOSI.TTF",
    "C:/Windows/Fonts/CENTURY.TTF",
    "C:/Windows/Fonts/BASKVILL.TTF",
    "C:/Windows/Fonts/GARA.TTF",
    "C:/Windows/Fonts/ARIALN.TTF",
)


@dataclass(frozen=True, slots=True)
class CaptchaVisualProfile:
    """Pixel-level ranges for one family of real captcha appearances."""

    name: VisualStyle
    fill_gray: tuple[int, int]
    stroke_gray: tuple[int, int]
    stroke_width: tuple[int, int]
    background_gray: tuple[int, int]
    background_noise: tuple[float, float]
    background_noise_gray: tuple[int, int]
    foreground_noise: tuple[float, float]
    foreground_noise_gray: tuple[int, int]
    scratches: tuple[int, int]

    def __post_init__(self) -> None:
        for value_range, maximum, label in (
            (self.fill_gray, 255, "fill gray"),
            (self.stroke_gray, 255, "stroke gray"),
            (self.background_gray, 255, "background gray"),
            (self.background_noise_gray, 255, "background noise gray"),
            (self.foreground_noise_gray, 255, "foreground noise gray"),
        ):
            if not 0 <= value_range[0] <= value_range[1] <= maximum:
                raise ValueError(f"{label} range is invalid")
        if not 0 <= self.stroke_width[0] <= self.stroke_width[1]:
            raise ValueError("stroke width range is invalid")
        if not 0 <= self.scratches[0] <= self.scratches[1]:
            raise ValueError("scratch range is invalid")
        for density_range in (self.background_noise, self.foreground_noise):
            if not 0 <= density_range[0] <= density_range[1] <= 1:
                raise ValueError("noise density range is invalid")


DEFAULT_VISUAL_PROFILES: dict[VisualStyle, CaptchaVisualProfile] = {
    "clean_outline": CaptchaVisualProfile(
        name="clean_outline",
        fill_gray=(245, 255),
        stroke_gray=(0, 25),
        stroke_width=(1, 2),
        background_gray=(252, 255),
        background_noise=(0.0, 0.0015),
        background_noise_gray=(145, 232),
        foreground_noise=(0.0, 0.002),
        foreground_noise_gray=(20, 175),
        scratches=(0, 1),
    ),
    "noisy_outline": CaptchaVisualProfile(
        name="noisy_outline",
        fill_gray=(200, 242),
        stroke_gray=(0, 35),
        stroke_width=(1, 1),
        background_gray=(248, 255),
        background_noise=(0.10, 0.18),
        background_noise_gray=(175, 225),
        foreground_noise=(0.025, 0.060),
        foreground_noise_gray=(135, 210),
        scratches=(5, 15),
    ),
    "solid": CaptchaVisualProfile(
        name="solid",
        fill_gray=(0, 22),
        stroke_gray=(0, 20),
        stroke_width=(1, 2),
        background_gray=(252, 255),
        background_noise=(0.0, 0.001),
        background_noise_gray=(145, 232),
        foreground_noise=(0.0, 0.001),
        foreground_noise_gray=(20, 175),
        scratches=(0, 0),
    ),
}


@dataclass(frozen=True, slots=True)
class ProjectCaptchaStyle:
    """Geometry and sampling controls for the 200 x 50 project captcha."""

    width: int = 200
    height: int = 50
    length: int = 4
    alphabet: str = PROJECT_ALPHABET
    font_path: str | None = None
    font_candidates: tuple[str, ...] = ()
    font_size: int = 63
    font_size_jitter: int = 5
    horizontal_scale: float = 1.45
    horizontal_scale_jitter: float = 0.25
    independent_horizontal_scale: bool = True
    vertical_scale_jitter: float = 0.10
    shear_degrees: float = 6.0
    target_width_min: float = 0.93
    target_width_max: float = 1.15
    rotation_degrees: float = 10.0
    vertical_jitter: int = 4
    horizontal_shift_max: int = 14
    overlap_min: int = 3
    overlap_max: int = 22
    clean_outline_weight: float = 75.0
    noisy_outline_weight: float = 37.0
    solid_weight: float = 26.0

    # Optional global overrides keep the preview CLI useful while profiles are tuned.
    background_noise_density: float | None = None
    foreground_noise_density: float | None = None
    scratch_count: int | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if not self.alphabet or len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet must be non-empty and cannot contain duplicates")
        if self.font_size <= 0 or self.font_size_jitter < 0:
            raise ValueError("font size must be positive and jitter cannot be negative")
        if self.font_size - self.font_size_jitter <= 0:
            raise ValueError("font size jitter produces a non-positive size")
        if self.horizontal_scale <= 0 or self.horizontal_scale_jitter < 0:
            raise ValueError("horizontal scale must be positive and jitter cannot be negative")
        if self.horizontal_scale - self.horizontal_scale_jitter <= 0:
            raise ValueError("horizontal scale jitter produces a non-positive scale")
        if not 0 <= self.vertical_scale_jitter < 1:
            raise ValueError("vertical scale jitter must be between zero and one")
        if not 0 <= self.shear_degrees < 45:
            raise ValueError("shear degrees must be between zero and 45")
        if not 0 < self.target_width_min <= self.target_width_max:
            raise ValueError("target width range is invalid")
        if self.rotation_degrees < 0 or self.vertical_jitter < 0:
            raise ValueError("rotation and jitter cannot be negative")
        if self.horizontal_shift_max < 0:
            raise ValueError("horizontal shift cannot be negative")
        if not 0 <= self.overlap_min <= self.overlap_max:
            raise ValueError("overlap range is invalid")

        weights = self.style_weights
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("style weights must be non-negative with a positive total")
        for density in (self.background_noise_density, self.foreground_noise_density):
            if density is not None and not 0 <= density <= 1:
                raise ValueError("noise density must be between 0 and 1")
        if self.scratch_count is not None and self.scratch_count < 0:
            raise ValueError("scratch count cannot be negative")

    @property
    def style_weights(self) -> tuple[float, float, float]:
        """Return weights in the same order as ``VISUAL_STYLES``."""

        return (
            self.clean_outline_weight,
            self.noisy_outline_weight,
            self.solid_weight,
        )


def project_style_for_geometry(preset: GeometryPreset) -> ProjectCaptchaStyle:
    """Build one of the geometry variants used by the v5 ablation experiment."""

    if preset == "enhanced":
        return ProjectCaptchaStyle()
    if preset == "classic":
        return ProjectCaptchaStyle(
            horizontal_scale_jitter=0.15,
            independent_horizontal_scale=False,
            vertical_scale_jitter=0.0,
            shear_degrees=0.0,
            target_width_max=1.12,
            rotation_degrees=8.0,
            vertical_jitter=3,
            horizontal_shift_max=12,
            overlap_min=6,
            overlap_max=16,
        )
    raise ValueError(f"unknown geometry preset: {preset}")


def project_style_for_source(
    preset: SourcePreset,
    geometry: GeometryPreset = "enhanced",
) -> ProjectCaptchaStyle:
    """Build a reproducible legacy or BotDetect-inspired source preset."""

    style = project_style_for_geometry(geometry)
    if preset == "legacy":
        return style
    if preset == "botdetect":
        return replace(style, font_candidates=BOTDETECT_FONT_CANDIDATES)
    raise ValueError(f"unknown source preset: {preset}")


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


def _resolve_font_paths(explicit_path: str | None, candidates: tuple[str, ...]) -> tuple[str, ...]:
    requested = ((explicit_path,) if explicit_path else ()) + candidates
    resolved: list[str] = []
    for candidate in requested:
        path = Path(candidate).expanduser()
        if path.is_file():
            value = str(path.resolve())
        else:
            try:
                ImageFont.truetype(candidate, 12)
            except OSError:
                continue
            value = candidate
        if value not in resolved:
            resolved.append(value)

    if not resolved:
        resolved.append(_resolve_font_path(explicit_path))
    return tuple(resolved)


class ProjectCaptchaGenerator:
    """Generate four-character captchas using observed project style frequencies."""

    def __init__(
        self,
        style: ProjectCaptchaStyle | None = None,
        profiles: dict[VisualStyle, CaptchaVisualProfile] | None = None,
    ) -> None:
        self.style = style or ProjectCaptchaStyle()
        self.font_paths = _resolve_font_paths(
            self.style.font_path,
            self.style.font_candidates,
        )
        # Kept for compatibility with earlier lessons and preview output.
        self.font_path = self.font_paths[0]
        self.profiles = dict(profiles or DEFAULT_VISUAL_PROFILES)
        missing = set(VISUAL_STYLES) - set(self.profiles)
        if missing:
            raise ValueError(f"missing visual profiles: {sorted(missing)}")

    def choose_visual_style(self, rng: random.Random) -> VisualStyle:
        """Sample one profile using weights estimated from 138 real images."""

        selected = rng.choices(VISUAL_STYLES, weights=self.style.style_weights, k=1)[0]
        return cast(VisualStyle, selected)

    def _render_character(
        self,
        character: str,
        profile: CaptchaVisualProfile,
        font: ImageFont.FreeTypeFont,
        horizontal_scale: float,
        vertical_scale: float,
        rng: random.Random,
    ) -> Image.Image:
        stroke = rng.randint(*profile.stroke_width)
        bbox = font.getbbox(character, stroke_width=stroke)
        padding = max(4, stroke + 3)
        width = bbox[2] - bbox[0] + padding * 2
        height = bbox[3] - bbox[1] + padding * 2
        glyph = Image.new("RGBA", (width, height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(glyph)

        fill_gray = rng.randint(*profile.fill_gray)
        stroke_gray = rng.randint(*profile.stroke_gray)
        draw.text(
            (padding - bbox[0], padding - bbox[1]),
            character,
            font=font,
            fill=(fill_gray, fill_gray, fill_gray, 255),
            stroke_width=stroke,
            stroke_fill=(stroke_gray, stroke_gray, stroke_gray, 255),
        )

        scaled_width = max(1, round(glyph.width * horizontal_scale))
        scaled_height = max(1, round(glyph.height * vertical_scale))
        glyph = glyph.resize((scaled_width, scaled_height), Image.Resampling.BICUBIC)

        if self.style.shear_degrees:
            shear = np.tan(
                np.deg2rad(rng.uniform(-self.style.shear_degrees, self.style.shear_degrees))
            )
            shear_extent = shear * glyph.height
            shear_offset = max(0.0, -shear_extent)
            sheared_width = max(1, glyph.width + round(abs(shear_extent)))
            glyph = glyph.transform(
                (sheared_width, glyph.height),
                Image.Transform.AFFINE,
                (1.0, -shear, -shear_offset, 0.0, 1.0, 0.0),
                resample=Image.Resampling.BICUBIC,
                fillcolor=(255, 255, 255, 0),
            )
        angle = rng.uniform(-self.style.rotation_degrees, self.style.rotation_degrees)
        glyph = glyph.rotate(
            angle,
            resample=Image.Resampling.BICUBIC,
            expand=True,
            fillcolor=(255, 255, 255, 0),
        )
        visible_bbox = glyph.getchannel("A").getbbox()
        return glyph.crop(visible_bbox) if visible_bbox else glyph

    def _place_characters(
        self,
        canvas: Image.Image,
        coverage: Image.Image,
        label: str,
        profile: CaptchaVisualProfile,
        rng: random.Random,
    ) -> None:
        font_size = rng.randint(
            self.style.font_size - self.style.font_size_jitter,
            self.style.font_size + self.style.font_size_jitter,
        )
        font = ImageFont.truetype(rng.choice(self.font_paths), font_size)
        if self.style.independent_horizontal_scale:
            horizontal_scales = [
                rng.uniform(
                    self.style.horizontal_scale - self.style.horizontal_scale_jitter,
                    self.style.horizontal_scale + self.style.horizontal_scale_jitter,
                )
                for _ in label
            ]
        else:
            shared_scale = rng.uniform(
                self.style.horizontal_scale - self.style.horizontal_scale_jitter,
                self.style.horizontal_scale + self.style.horizontal_scale_jitter,
            )
            horizontal_scales = [shared_scale] * len(label)
        vertical_scales = (
            [
                rng.uniform(
                    1.0 - self.style.vertical_scale_jitter,
                    1.0 + self.style.vertical_scale_jitter,
                )
                for _ in label
            ]
            if self.style.vertical_scale_jitter
            else [1.0] * len(label)
        )
        glyphs = [
            self._render_character(
                character,
                profile,
                font,
                horizontal_scale,
                vertical_scale,
                rng,
            )
            for character, horizontal_scale, vertical_scale in zip(
                label, horizontal_scales, vertical_scales, strict=True
            )
        ]
        overlaps = [
            rng.randint(self.style.overlap_min, self.style.overlap_max)
            for _ in range(len(glyphs) - 1)
        ]

        total_width = sum(glyph.width for glyph in glyphs) - sum(overlaps)
        target_width = round(
            self.style.width * rng.uniform(self.style.target_width_min, self.style.target_width_max)
        )
        width_scale = target_width / total_width
        glyphs = [
            glyph.resize(
                (max(1, round(glyph.width * width_scale)), glyph.height),
                Image.Resampling.BICUBIC,
            )
            for glyph in glyphs
        ]
        overlaps = [max(1, round(overlap * width_scale)) for overlap in overlaps]
        total_width = sum(glyph.width for glyph in glyphs) - sum(overlaps)

        centered_x = (self.style.width - total_width) // 2
        x = max(0, centered_x + rng.randint(0, self.style.horizontal_shift_max))
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
        profile: CaptchaVisualProfile,
        rng: random.Random,
    ) -> None:
        draw = ImageDraw.Draw(canvas)
        mask = np.asarray(coverage)
        background_points = np.argwhere(mask < 16)
        foreground_points = np.argwhere(mask > 64)

        background_density = self.style.background_noise_density
        if background_density is None:
            background_density = rng.uniform(*profile.background_noise)
        background_count = round(self.style.width * self.style.height * background_density)
        for _ in range(background_count):
            if not len(background_points):
                break
            y, x = background_points[rng.randrange(len(background_points))]
            gray = rng.randint(*profile.background_noise_gray)
            radius = 1 if rng.random() < 0.12 else 0
            draw.ellipse(
                (int(x) - radius, int(y) - radius, int(x) + radius, int(y) + radius),
                fill=gray,
            )

        foreground_density = self.style.foreground_noise_density
        if foreground_density is None:
            foreground_density = rng.uniform(*profile.foreground_noise)
        foreground_count = round(len(foreground_points) * foreground_density)
        for _ in range(foreground_count):
            if not len(foreground_points):
                break
            y, x = foreground_points[rng.randrange(len(foreground_points))]
            canvas.putpixel((int(x), int(y)), rng.randint(*profile.foreground_noise_gray))

        scratch_count = self.style.scratch_count
        if scratch_count is None:
            scratch_count = rng.randint(*profile.scratches)
        for _ in range(scratch_count):
            x = rng.randrange(self.style.width)
            y = rng.randrange(self.style.height)
            length = rng.randint(1, 4)
            end_y = min(self.style.height - 1, max(0, y + rng.choice((-1, 0, 1))))
            draw.line(
                (x, y, min(self.style.width - 1, x + length), end_y),
                fill=rng.randint(160, 232),
                width=1,
            )

    def generate(
        self,
        text: str | None = None,
        *,
        rng: random.Random | None = None,
        visual_style: VisualStyle | None = None,
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
        if visual_style is not None and visual_style not in VISUAL_STYLES:
            raise ValueError(f"unknown visual style: {visual_style}")

        selected_style = visual_style or self.choose_visual_style(source)
        profile = self.profiles[selected_style]
        background_gray = source.randint(*profile.background_gray)
        canvas = Image.new("L", (self.style.width, self.style.height), background_gray)
        coverage = Image.new("L", canvas.size, 0)
        self._place_characters(canvas, coverage, label, profile, source)
        self._add_noise(canvas, coverage, profile, source)
        return canvas, label
