import random

import pytest

from captcha_break.project_generator import ProjectCaptchaGenerator, ProjectCaptchaStyle


def test_project_generator_returns_grayscale_200_by_50_image() -> None:
    generator = ProjectCaptchaGenerator(ProjectCaptchaStyle())
    image, label = generator.generate("KJUU", rng=random.Random(7))

    assert image.size == (200, 50)
    assert image.mode == "L"
    assert label == "KJUU"


def test_project_generator_is_repeatable_with_seeded_rng() -> None:
    generator = ProjectCaptchaGenerator(ProjectCaptchaStyle())
    first, _ = generator.generate("A7K2", rng=random.Random(42))
    second, _ = generator.generate("A7K2", rng=random.Random(42))

    assert first.tobytes() == second.tobytes()


def test_project_generator_rejects_wrong_length() -> None:
    generator = ProjectCaptchaGenerator(ProjectCaptchaStyle())
    with pytest.raises(ValueError, match="exactly 4"):
        generator.generate("ABC")
