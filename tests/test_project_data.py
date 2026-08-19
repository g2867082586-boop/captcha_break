from __future__ import annotations

import random

import torch
from PIL import Image

from captcha_break.codec import decode_indices
from captcha_break.config import DEFAULT_ALPHABET
from captcha_break.data import (
    ProjectCaptchaDataset,
    RealCaptchaDataset,
    TargetedLabelSampling,
    grayscale_image_to_tensor,
    label_from_filename,
)


def test_grayscale_image_to_tensor_returns_one_channel() -> None:
    image = Image.new("RGB", (200, 50), (128, 128, 128))

    tensor = grayscale_image_to_tensor(image)

    assert tensor.shape == (1, 50, 200)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor.mean(), torch.tensor(128 / 255))


def test_project_dataset_returns_repeatable_seeded_sample() -> None:
    dataset = ProjectCaptchaDataset(size=3, seed=2026)

    first_image, first_target = dataset[0]
    repeated_image, repeated_target = dataset[0]

    assert first_image.shape == (1, 50, 200)
    assert first_target.shape == (4,)
    assert first_target.dtype == torch.long
    assert torch.equal(first_image, repeated_image)
    assert torch.equal(first_target, repeated_target)
    assert len(decode_indices(first_target.tolist(), dataset.characters)) == 4


def test_real_dataset_reads_label_from_filename(tmp_path) -> None:
    Image.new("RGB", (200, 50), "white").save(tmp_path / "A7K2_001.png")
    dataset = RealCaptchaDataset(tmp_path, characters=DEFAULT_ALPHABET)

    image, target = dataset[0]

    assert len(dataset) == 1
    assert image.shape == (1, 50, 200)
    assert decode_indices(target.tolist(), dataset.characters) == "A7K2"


def test_real_dataset_rejects_wrong_image_size(tmp_path) -> None:
    Image.new("RGB", (100, 30), "white").save(tmp_path / "B8M3_001.png")
    dataset = RealCaptchaDataset(tmp_path, characters=DEFAULT_ALPHABET)

    try:
        dataset[0]
    except ValueError as error:
        assert "expected image size" in str(error)
    else:
        raise AssertionError("wrong-sized real image should be rejected")


def test_label_from_filename_validates_characters() -> None:
    assert label_from_filename("DYJX_004.jfif", DEFAULT_ALPHABET) == "DYJX"


def test_targeted_sampling_always_places_a_target_at_weighted_position() -> None:
    sampling = TargetedLabelSampling(
        characters="MU",
        probability=1.0,
        position_weights=(0.0, 1.0, 0.0, 0.0),
    )

    label = sampling.sample(DEFAULT_ALPHABET, 4, random.Random(7))

    assert label[1] in "MU"


def test_project_dataset_accepts_targeted_sampling() -> None:
    sampling = TargetedLabelSampling(
        characters="M",
        probability=1.0,
        position_weights=(0.0, 1.0, 0.0, 0.0),
    )
    dataset = ProjectCaptchaDataset(size=1, seed=17, label_sampling=sampling)

    _, target = dataset[0]

    assert target[1].item() == dataset.characters.index("M")
