"""Shared project configuration."""

from __future__ import annotations

import string
from dataclasses import dataclass

DEFAULT_ALPHABET = string.digits + string.ascii_uppercase
CTC_BLANK = "-"


@dataclass(frozen=True, slots=True)
class CaptchaConfig:
    """Image and label settings used by generation and recognition."""

    width: int = 192
    height: int = 64
    length: int = 4
    alphabet: str = DEFAULT_ALPHABET

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if len(self.alphabet) < 2:
            raise ValueError("alphabet must contain at least two characters")
        if len(set(self.alphabet)) != len(self.alphabet):
            raise ValueError("alphabet cannot contain duplicate characters")
        if CTC_BLANK in self.alphabet:
            raise ValueError(f"alphabet cannot contain the CTC blank character {CTC_BLANK!r}")

    @property
    def ctc_characters(self) -> str:
        """Characters used by CTC; index 0 is reserved for the blank token."""

        return CTC_BLANK + self.alphabet
