import random

import pytest

from captcha_break import CaptchaConfig, generate_captcha


def test_generate_captcha_has_expected_size_and_label() -> None:
    config = CaptchaConfig(width=160, height=60, length=4, alphabet="AB12")
    image, label = generate_captcha(config, rng=random.Random(7))
    assert image.size == (160, 60)
    assert image.mode == "RGB"
    assert len(label) == 4
    assert set(label) <= set(config.alphabet)


def test_invalid_fixed_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 4"):
        generate_captcha(text="ABC")
    with pytest.raises(ValueError, match="exactly 4"):
        generate_captcha(text="")
