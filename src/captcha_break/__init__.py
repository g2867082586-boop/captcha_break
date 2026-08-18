"""Captcha generation and recognition helpers."""

from .config import DEFAULT_ALPHABET, CaptchaConfig
from .generator import generate_captcha, random_text
from .project_generator import ProjectCaptchaGenerator, ProjectCaptchaStyle

__all__ = [
    "DEFAULT_ALPHABET",
    "CaptchaConfig",
    "ProjectCaptchaGenerator",
    "ProjectCaptchaStyle",
    "generate_captcha",
    "random_text",
]
__version__ = "0.1.0"
