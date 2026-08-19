"""Captcha generation and recognition helpers."""

from .config import DEFAULT_ALPHABET, CaptchaConfig
from .generator import generate_captcha, random_text
from .project_generator import (
    BOTDETECT_FONT_CANDIDATES,
    BOTDETECT_STYLE_NAMES,
    DEFAULT_VISUAL_PROFILES,
    PROJECT_ALPHABET,
    VISUAL_STYLES,
    CaptchaVisualProfile,
    GeometryPreset,
    ProjectCaptchaGenerator,
    ProjectCaptchaStyle,
    SourcePreset,
    VisualStyle,
    project_style_for_geometry,
    project_style_for_source,
)

__all__ = [
    "BOTDETECT_FONT_CANDIDATES",
    "BOTDETECT_STYLE_NAMES",
    "DEFAULT_ALPHABET",
    "DEFAULT_VISUAL_PROFILES",
    "PROJECT_ALPHABET",
    "VISUAL_STYLES",
    "CaptchaConfig",
    "CaptchaVisualProfile",
    "GeometryPreset",
    "ProjectCaptchaGenerator",
    "ProjectCaptchaStyle",
    "SourcePreset",
    "VisualStyle",
    "generate_captcha",
    "project_style_for_geometry",
    "project_style_for_source",
    "random_text",
]
__version__ = "0.1.0"
