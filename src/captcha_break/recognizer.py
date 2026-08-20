"""Stable application-facing interface for the project's CAPTCHA recognizer."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from .ddddocr_adapter import DdddOcrFallbackRecognizer, FallbackPrediction
from .project_generator import PROJECT_ALPHABET

ImageSource = bytes | bytearray | str | Path


class DetailedRecognizer(Protocol):
    """Interface required from the internal recognition engine."""

    def predict_detailed(self, image: bytes) -> FallbackPrediction: ...


@dataclass(frozen=True, slots=True)
class RecognitionResult:
    """JSON-friendly result returned to the application layer."""

    text: str
    model_used: str
    milliseconds: float
    is_structurally_valid: bool
    used_fallback: bool
    beta_text: str
    default_text: str | None
    fallback_reason: str | None

    @property
    def status(self) -> str:
        """Describe structural validation without claiming OCR certainty."""

        return "structurally_valid" if self.is_structurally_valid else "needs_review"

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["milliseconds"] = round(self.milliseconds, 3)
        result["status"] = self.status
        return result


@dataclass(frozen=True, slots=True)
class BatchRecognitionItem:
    """One successful or failed item in a batch recognition request."""

    identifier: str
    result: RecognitionResult | None
    error: str | None

    @property
    def succeeded(self) -> bool:
        return self.result is not None

    def as_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "identifier": self.identifier,
            "succeeded": self.succeeded,
            "error": self.error,
        }
        if self.result is not None:
            row.update(self.result.as_dict())
        return row


def image_source_to_bytes(image: ImageSource) -> bytes:
    """Convert a local path, byte buffer, or base64 data URL into image bytes."""

    if isinstance(image, bytes):
        if not image:
            raise ValueError("image bytes cannot be empty")
        return image
    if isinstance(image, bytearray):
        if not image:
            raise ValueError("image bytes cannot be empty")
        return bytes(image)
    if isinstance(image, Path):
        path = image.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"image file does not exist: {path}")
        return path.read_bytes()
    if not isinstance(image, str):
        raise TypeError(f"unsupported image source: {type(image).__name__}")

    if image.startswith("data:image/"):
        header, separator, encoded = image.partition(",")
        if not separator or ";base64" not in header.lower():
            raise ValueError("image data URL must contain base64 data")
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValueError("image data URL contains invalid base64 data") from error
        if not decoded:
            raise ValueError("decoded image data cannot be empty")
        return decoded

    path = Path(image).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image file does not exist: {path}")
    return path.read_bytes()


class ProjectCaptchaRecognizer:
    """Reusable recognizer for one 4-character project CAPTCHA at a time."""

    def __init__(self, engine: DetailedRecognizer | None = None) -> None:
        self.engine = engine or DdddOcrFallbackRecognizer(
            expected_length=4,
            alphabet=PROJECT_ALPHABET,
        )

    def recognize(self, image: ImageSource) -> RecognitionResult:
        """Recognize a supported image source and return auditable metadata."""

        result = self.engine.predict_detailed(image_source_to_bytes(image))
        return RecognitionResult(
            text=result.text,
            model_used=result.model_used,
            milliseconds=result.milliseconds,
            is_structurally_valid=result.is_valid,
            used_fallback=result.used_fallback,
            beta_text=result.beta_text,
            default_text=result.default_text,
            fallback_reason=result.fallback_reason,
        )

    def recognize_text(self, image: ImageSource) -> str:
        """Convenience method for callers that only need the recognized text."""

        return self.recognize(image).text

    def recognize_many(
        self,
        images: Iterable[tuple[str, ImageSource]],
        *,
        continue_on_error: bool = True,
    ) -> list[BatchRecognitionItem]:
        """Recognize named inputs while optionally isolating individual failures."""

        items: list[BatchRecognitionItem] = []
        for identifier, image in images:
            if not identifier:
                raise ValueError("batch item identifier cannot be empty")
            try:
                result = self.recognize(image)
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                if not continue_on_error:
                    raise
                items.append(
                    BatchRecognitionItem(
                        identifier=identifier,
                        result=None,
                        error=f"{type(error).__name__}: {error}",
                    )
                )
            else:
                items.append(
                    BatchRecognitionItem(
                        identifier=identifier,
                        result=result,
                        error=None,
                    )
                )
        return items
