import random

import pytest
from PIL import ImageStat

from captcha_break.project_generator import (
    BOTDETECT_FONT_CANDIDATES,
    BOTDETECT_STYLE_NAMES,
    PROJECT_ALPHABET,
    VISUAL_STYLES,
    ProjectCaptchaGenerator,
    ProjectCaptchaStyle,
    project_style_for_geometry,
    project_style_for_source,
)


def test_project_generator_returns_grayscale_200_by_50_image() -> None:
    generator = ProjectCaptchaGenerator(ProjectCaptchaStyle())
    image, label = generator.generate("KJUU", rng=random.Random(7))

    assert image.size == (200, 50)
    assert image.mode == "L"
    assert label == "KJUU"


def test_project_generator_is_repeatable_with_seeded_rng() -> None:
    generator = ProjectCaptchaGenerator(ProjectCaptchaStyle())
    first, _ = generator.generate("A8K3", rng=random.Random(42))
    second, _ = generator.generate("A8K3", rng=random.Random(42))

    assert first.tobytes() == second.tobytes()


def test_project_generator_rejects_wrong_length() -> None:
    generator = ProjectCaptchaGenerator(ProjectCaptchaStyle())
    with pytest.raises(ValueError, match="exactly 4"):
        generator.generate("ABC")


def test_visual_styles_follow_observed_real_frequencies_by_default() -> None:
    style = ProjectCaptchaStyle()

    assert style.style_weights == (75.0, 37.0, 26.0)


def test_project_alphabet_matches_characters_observed_in_138_real_images() -> None:
    assert PROJECT_ALPHABET == "34689ABCDEHJKMNPRTUVWXY"
    assert len(PROJECT_ALPHABET) == 23


def test_visual_style_can_be_forced() -> None:
    generator = ProjectCaptchaGenerator()
    clean, _ = generator.generate("B938", rng=random.Random(11), visual_style="clean_outline")
    solid, _ = generator.generate("B938", rng=random.Random(11), visual_style="solid")

    assert ImageStat.Stat(solid).mean[0] < ImageStat.Stat(clean).mean[0]


def test_zero_weights_select_the_only_enabled_style() -> None:
    generator = ProjectCaptchaGenerator(
        ProjectCaptchaStyle(
            clean_outline_weight=0.0,
            noisy_outline_weight=0.0,
            solid_weight=1.0,
        )
    )

    assert {generator.choose_visual_style(random.Random(seed)) for seed in range(20)} == {"solid"}


def test_generator_rejects_unknown_visual_style() -> None:
    generator = ProjectCaptchaGenerator()

    with pytest.raises(ValueError, match="unknown visual style"):
        generator.generate("KJUU", visual_style="unknown")  # type: ignore[arg-type]


def test_visual_style_names_are_stable() -> None:
    assert VISUAL_STYLES == ("clean_outline", "noisy_outline", "solid")


def test_generator_rejects_invalid_geometric_jitter() -> None:
    with pytest.raises(ValueError, match="vertical scale jitter"):
        ProjectCaptchaStyle(vertical_scale_jitter=1.0)
    with pytest.raises(ValueError, match="shear degrees"):
        ProjectCaptchaStyle(shear_degrees=45.0)


def test_geometry_presets_keep_ablation_variables_separate() -> None:
    classic = project_style_for_geometry("classic")
    enhanced = project_style_for_geometry("enhanced")

    assert classic.independent_horizontal_scale is False
    assert classic.vertical_scale_jitter == 0.0
    assert classic.shear_degrees == 0.0
    assert classic.overlap_max == 16
    assert enhanced.independent_horizontal_scale is True
    assert enhanced.vertical_scale_jitter == 0.10
    assert enhanced.shear_degrees == 6.0
    assert enhanced.overlap_max == 22


def test_botdetect_source_preset_adds_font_variation_without_changing_shape() -> None:
    legacy = project_style_for_source("legacy")
    botdetect = project_style_for_source("botdetect")

    assert legacy.font_candidates == ()
    assert botdetect.font_candidates == BOTDETECT_FONT_CANDIDATES
    assert botdetect.width == legacy.width == 200
    assert botdetect.height == legacy.height == 50
    assert botdetect.alphabet == legacy.alphabet


def test_observed_visual_families_have_botdetect_reference_names() -> None:
    assert BOTDETECT_STYLE_NAMES == {
        "clean_outline": "Overlap2",
        "noisy_outline": "Rough",
        "solid": "BlackOverlap",
    }
