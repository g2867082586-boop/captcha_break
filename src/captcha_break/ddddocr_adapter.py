"""Optional ddddocr baseline integration and fixed-label evaluation metrics."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:
    from ddddocr import DdddOcr


@dataclass(frozen=True, slots=True)
class OcrMetrics:
    """Accuracy counts for labeled OCR predictions."""

    image_count: int
    exact_count: int
    character_count: int
    correct_character_count: int
    matching_length_count: int

    @property
    def exact_accuracy(self) -> float:
        return self.exact_count / self.image_count

    @property
    def character_accuracy(self) -> float:
        return self.correct_character_count / self.character_count

    @property
    def matching_length_accuracy(self) -> float:
        return self.matching_length_count / self.image_count


def calculate_ocr_metrics(targets: Sequence[str], predictions: Sequence[str]) -> OcrMetrics:
    """Calculate positional character, exact-image, and output-length accuracy."""

    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must contain the same number of items")
    if not targets:
        raise ValueError("at least one target is required")
    if any(not target for target in targets):
        raise ValueError("targets cannot be empty")

    exact_count = sum(target == prediction for target, prediction in zip(targets, predictions))
    correct_character_count = sum(
        sum(expected == actual for expected, actual in zip(target, prediction))
        for target, prediction in zip(targets, predictions)
    )
    matching_length_count = sum(
        len(target) == len(prediction) for target, prediction in zip(targets, predictions)
    )
    return OcrMetrics(
        image_count=len(targets),
        exact_count=exact_count,
        character_count=sum(len(target) for target in targets),
        correct_character_count=correct_character_count,
        matching_length_count=matching_length_count,
    )


class DdddOcrRecognizer:
    """Small lazy-loading wrapper around the optional ddddocr dependency."""

    def __init__(self, *, beta: bool = True) -> None:
        try:
            import ddddocr
        except ImportError as error:
            raise RuntimeError(
                "ddddocr is not installed; run `uv sync --extra baseline` first"
            ) from error

        self.beta = beta
        self.engine: DdddOcr = ddddocr.DdddOcr(beta=beta, show_ad=False)

    @property
    def model_name(self) -> str:
        return "beta" if self.beta else "default"

    def predict(self, image: bytes | bytearray | str | Path) -> str:
        """Recognize one local image without post-filtering the character set."""

        result = self.engine.classification(image)
        if not isinstance(result, str):
            raise TypeError(f"ddddocr returned an unexpected result: {type(result).__name__}")
        return result.upper()


ImageInput = bytes | bytearray | str | Path
ModelName = Literal["beta", "default"]


class OcrRecognizer(Protocol):
    """Minimal interface used by the fallback recognizer and its tests."""

    def predict(self, image: ImageInput) -> str: ...


@dataclass(frozen=True, slots=True)
class FallbackPrediction:
    """One prediction plus the information needed to audit its selection."""

    text: str
    model_used: ModelName
    beta_text: str
    default_text: str | None
    fallback_reason: str | None
    is_valid: bool
    milliseconds: float

    @property
    def used_fallback(self) -> bool:
        return self.default_text is not None


class DdddOcrFallbackRecognizer:
    """Use Beta first and call Default only when Beta fails structural checks."""

    def __init__(
        self,
        *,
        expected_length: int,
        alphabet: str,
        beta_recognizer: OcrRecognizer | None = None,
        default_factory: Callable[[], OcrRecognizer] | None = None,
    ) -> None:
        if expected_length <= 0:
            raise ValueError("expected_length must be positive")
        if not alphabet:
            raise ValueError("alphabet cannot be empty")

        self.expected_length = expected_length
        self.alphabet = frozenset(alphabet.upper())
        self.beta_recognizer = beta_recognizer or DdddOcrRecognizer(beta=True)
        self.default_factory = default_factory or (
            lambda: DdddOcrRecognizer(beta=False)
        )
        self._default_recognizer: OcrRecognizer | None = None

    def _invalid_reason(self, text: str) -> str | None:
        if len(text) != self.expected_length:
            return f"length={len(text)}, expected={self.expected_length}"
        invalid_characters = sorted(set(text) - self.alphabet)
        if invalid_characters:
            return f"characters outside alphabet: {''.join(invalid_characters)}"
        return None

    def _get_default_recognizer(self) -> OcrRecognizer:
        if self._default_recognizer is None:
            self._default_recognizer = self.default_factory()
        return self._default_recognizer

    def predict_detailed(self, image: ImageInput) -> FallbackPrediction:
        """Return the chosen text and why the fallback was or was not used."""

        started = perf_counter()
        beta_text = self.beta_recognizer.predict(image).upper()
        beta_reason = self._invalid_reason(beta_text)
        if beta_reason is None:
            return FallbackPrediction(
                text=beta_text,
                model_used="beta",
                beta_text=beta_text,
                default_text=None,
                fallback_reason=None,
                is_valid=True,
                milliseconds=(perf_counter() - started) * 1000,
            )

        default_text = self._get_default_recognizer().predict(image).upper()
        default_is_valid = self._invalid_reason(default_text) is None
        return FallbackPrediction(
            text=default_text if default_is_valid else beta_text,
            model_used="default" if default_is_valid else "beta",
            beta_text=beta_text,
            default_text=default_text,
            fallback_reason=beta_reason,
            is_valid=default_is_valid,
            milliseconds=(perf_counter() - started) * 1000,
        )

    def predict(self, image: ImageInput) -> str:
        """Return only the selected text for application integration."""

        return self.predict_detailed(image).text
