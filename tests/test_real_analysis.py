from PIL import Image

from captcha_break.real_analysis import RealImageMetrics, classify_real_style, measure_image


def metrics(dark_ratio: float, foreground_ratio: float) -> RealImageMetrics:
    return RealImageMetrics(
        mean_gray=200.0,
        dark_ratio=dark_ratio,
        foreground_ratio=foreground_ratio,
        bbox=(0, 0, 200, 50),
    )


def test_classify_solid_style() -> None:
    assert classify_real_style(metrics(0.40, 0.45)) == "solid"


def test_classify_noisy_outline_style() -> None:
    assert classify_real_style(metrics(0.09, 0.26)) == "noisy_outline"


def test_classify_clean_outline_style() -> None:
    assert classify_real_style(metrics(0.20, 0.27)) == "clean_outline"


def test_measure_image_reports_full_foreground_bbox() -> None:
    image = Image.new("L", (200, 50), 255)
    for x in range(20, 180):
        image.putpixel((x, 25), 0)

    measured = measure_image(image)

    assert measured.bbox == (20, 25, 180, 26)
    assert measured.dark_ratio > 0
