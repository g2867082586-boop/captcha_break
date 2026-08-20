import pytest

from captcha_break.ddddocr_adapter import (
    DdddOcrFallbackRecognizer,
    calculate_ocr_metrics,
)


class FakeRecognizer:
    def __init__(self, prediction: str) -> None:
        self.prediction = prediction
        self.call_count = 0

    def predict(self, image: object) -> str:
        self.call_count += 1
        return self.prediction


def test_calculate_ocr_metrics_handles_missing_and_extra_characters() -> None:
    metrics = calculate_ocr_metrics(
        ["ABCD", "WXYZ", "3333"],
        ["ABCD", "WXY", "33333"],
    )

    assert metrics.image_count == 3
    assert metrics.exact_count == 1
    assert metrics.correct_character_count == 11
    assert metrics.character_count == 12
    assert metrics.matching_length_count == 1
    assert metrics.exact_accuracy == pytest.approx(1 / 3)
    assert metrics.character_accuracy == pytest.approx(11 / 12)
    assert metrics.matching_length_accuracy == pytest.approx(1 / 3)


def test_calculate_ocr_metrics_validates_inputs() -> None:
    with pytest.raises(ValueError, match="same number"):
        calculate_ocr_metrics(["ABCD"], [])
    with pytest.raises(ValueError, match="at least one"):
        calculate_ocr_metrics([], [])
    with pytest.raises(ValueError, match="cannot be empty"):
        calculate_ocr_metrics([""], [""])


def test_fallback_recognizer_keeps_valid_beta_without_loading_default() -> None:
    beta = FakeRecognizer("ab34")
    default_factory_calls = 0

    def create_default() -> FakeRecognizer:
        nonlocal default_factory_calls
        default_factory_calls += 1
        return FakeRecognizer("ABCD")

    recognizer = DdddOcrFallbackRecognizer(
        expected_length=4,
        alphabet="ABCD34",
        beta_recognizer=beta,
        default_factory=create_default,
    )

    result = recognizer.predict_detailed(b"image")

    assert result.text == "AB34"
    assert result.model_used == "beta"
    assert result.default_text is None
    assert not result.used_fallback
    assert result.is_valid
    assert default_factory_calls == 0


@pytest.mark.parametrize("beta_text", ["ABC", "AB?D"])
def test_fallback_recognizer_uses_valid_default_for_invalid_beta(
    beta_text: str,
) -> None:
    default = FakeRecognizer("ABCD")
    recognizer = DdddOcrFallbackRecognizer(
        expected_length=4,
        alphabet="ABCD",
        beta_recognizer=FakeRecognizer(beta_text),
        default_factory=lambda: default,
    )

    result = recognizer.predict_detailed(b"image")

    assert result.text == "ABCD"
    assert result.model_used == "default"
    assert result.default_text == "ABCD"
    assert result.used_fallback
    assert result.fallback_reason is not None
    assert result.is_valid
    assert default.call_count == 1


def test_fallback_recognizer_flags_result_when_both_models_are_invalid() -> None:
    recognizer = DdddOcrFallbackRecognizer(
        expected_length=4,
        alphabet="ABCD",
        beta_recognizer=FakeRecognizer("ABC"),
        default_factory=lambda: FakeRecognizer("ABCDE"),
    )

    result = recognizer.predict_detailed(b"image")

    assert result.text == "ABC"
    assert result.model_used == "beta"
    assert result.beta_text == "ABC"
    assert result.default_text == "ABCDE"
    assert result.used_fallback
    assert not result.is_valid


@pytest.mark.parametrize(
    ("expected_length", "alphabet", "message"),
    [(0, "ABCD", "positive"), (4, "", "cannot be empty")],
)
def test_fallback_recognizer_validates_configuration(
    expected_length: int,
    alphabet: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        DdddOcrFallbackRecognizer(
            expected_length=expected_length,
            alphabet=alphabet,
            beta_recognizer=FakeRecognizer("ABCD"),
        )
