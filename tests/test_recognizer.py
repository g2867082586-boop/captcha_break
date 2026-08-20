import base64
from pathlib import Path

import pytest

from captcha_break.ddddocr_adapter import FallbackPrediction
from captcha_break.recognizer import ProjectCaptchaRecognizer, image_source_to_bytes


class FakeDetailedRecognizer:
    def __init__(self, result: FallbackPrediction) -> None:
        self.result = result
        self.received: bytes | None = None

    def predict_detailed(self, image: bytes) -> FallbackPrediction:
        self.received = image
        return self.result


class ConditionalDetailedRecognizer(FakeDetailedRecognizer):
    def predict_detailed(self, image: bytes) -> FallbackPrediction:
        if image == b"bad":
            raise ValueError("broken image")
        return super().predict_detailed(image)


def test_image_source_to_bytes_supports_bytes_bytearray_and_path(tmp_path: Path) -> None:
    image_path = tmp_path / "captcha.jpg"
    image_path.write_bytes(b"jpeg-data")

    assert image_source_to_bytes(b"raw") == b"raw"
    assert image_source_to_bytes(bytearray(b"buffer")) == b"buffer"
    assert image_source_to_bytes(image_path) == b"jpeg-data"
    assert image_source_to_bytes(str(image_path)) == b"jpeg-data"


def test_image_source_to_bytes_decodes_base64_data_url() -> None:
    encoded = base64.b64encode(b"jpeg-data").decode("ascii")

    assert image_source_to_bytes(f"data:image/jpeg;base64,{encoded}") == b"jpeg-data"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (b"", "cannot be empty"),
        ("data:image/jpeg,abc", "must contain base64"),
        ("data:image/jpeg;base64,%%%", "invalid base64"),
    ],
)
def test_image_source_to_bytes_rejects_invalid_input(
    source: bytes | str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        image_source_to_bytes(source)


def test_project_recognizer_returns_application_friendly_result() -> None:
    engine = FakeDetailedRecognizer(
        FallbackPrediction(
            text="KJUU",
            model_used="beta",
            beta_text="KJUU",
            default_text=None,
            fallback_reason=None,
            is_valid=True,
            milliseconds=12.5,
        )
    )
    recognizer = ProjectCaptchaRecognizer(engine=engine)

    result = recognizer.recognize(b"image")

    assert engine.received == b"image"
    assert result.text == "KJUU"
    assert result.model_used == "beta"
    assert result.status == "structurally_valid"
    assert result.as_dict()["status"] == "structurally_valid"
    assert recognizer.recognize_text(b"image") == "KJUU"


def test_project_recognizer_isolates_batch_item_failures() -> None:
    engine = ConditionalDetailedRecognizer(
        FallbackPrediction(
            text="KJUU",
            model_used="beta",
            beta_text="KJUU",
            default_text=None,
            fallback_reason=None,
            is_valid=True,
            milliseconds=12.5,
        )
    )
    recognizer = ProjectCaptchaRecognizer(engine=engine)

    items = recognizer.recognize_many(
        [("good.jpg", b"good"), ("bad.jpg", b"bad")]
    )

    assert len(items) == 2
    assert items[0].succeeded
    assert items[0].result is not None
    assert items[0].result.text == "KJUU"
    assert not items[1].succeeded
    assert items[1].error == "ValueError: broken image"


def test_project_recognizer_can_stop_batch_on_first_error() -> None:
    engine = ConditionalDetailedRecognizer(
        FallbackPrediction(
            text="KJUU",
            model_used="beta",
            beta_text="KJUU",
            default_text=None,
            fallback_reason=None,
            is_valid=True,
            milliseconds=12.5,
        )
    )
    recognizer = ProjectCaptchaRecognizer(engine=engine)

    with pytest.raises(ValueError, match="broken image"):
        recognizer.recognize_many(
            [("bad.jpg", b"bad")],
            continue_on_error=False,
        )
