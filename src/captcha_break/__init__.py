"""Captcha generation and recognition helpers."""

from .config import DEFAULT_ALPHABET, CaptchaConfig
from .generator import generate_captcha, random_text
from .project_generator import (
    DEFAULT_VISUAL_PROFILES,
    PROJECT_ALPHABET,
    VISUAL_STYLES,
    CaptchaVisualProfile,
    GeometryPreset,
    ProjectCaptchaGenerator,
    ProjectCaptchaStyle,
    VisualStyle,
    project_style_for_geometry,
)

__all__ = [
    "DEFAULT_ALPHABET",
    "DEFAULT_VISUAL_PROFILES",
    "PROJECT_ALPHABET",
    "VISUAL_STYLES",
    "CaptchaConfig",
    "CaptchaVisualProfile",
    "GeometryPreset",
    "ProjectCaptchaGenerator",
    "ProjectCaptchaStyle",
    "VisualStyle",
    "generate_captcha",
    "project_style_for_geometry",
    "random_text",
]
__version__ = "0.1.0"
