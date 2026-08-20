from pathlib import Path

import pytest
from PIL import Image

from captcha_break.dataset_split import (
    copy_normalized_captcha,
    make_three_way_split,
)
from captcha_break.project_generator import PROJECT_ALPHABET


def test_three_way_split_is_disjoint_and_repeatable(tmp_path: Path) -> None:
    labels = [PROJECT_ALPHABET[index % len(PROJECT_ALPHABET)] * 4 for index in range(30)]
    paths = [tmp_path / f"{label}_{index:04d}.png" for index, label in enumerate(labels)]
    styles = {
        path: ("clean_outline" if index % 2 else "noisy_outline")
        for index, path in enumerate(paths)
    }
    first = make_three_way_split(
        paths,
        styles,
        train_size=20,
        validation_size=5,
        test_size=5,
        alphabet=PROJECT_ALPHABET,
        seed=2039,
        candidates=20,
    )
    second = make_three_way_split(
        paths,
        styles,
        train_size=20,
        validation_size=5,
        test_size=5,
        alphabet=PROJECT_ALPHABET,
        seed=2039,
        candidates=20,
    )
    assert first == second
    assert len(first.train) == 20
    assert len(first.validation) == 5
    assert len(first.test) == 5
    assert not (set(first.train) & set(first.validation))
    assert not (set(first.train) & set(first.test))
    assert not (set(first.validation) & set(first.test))


def test_copy_normalized_captcha_removes_known_white_padding(tmp_path: Path) -> None:
    source = tmp_path / "AAAA_0001.png"
    destination = tmp_path / "output" / source.name
    image = Image.new("L", (201, 51), 255)
    image.putpixel((10, 10), 0)
    image.save(source)
    assert copy_normalized_captcha(source, destination) == "cropped"
    with Image.open(destination) as normalized:
        assert normalized.size == (200, 50)
        assert normalized.getpixel((10, 10)) == 0


def test_copy_normalized_captcha_rejects_nonwhite_extra_edge(tmp_path: Path) -> None:
    source = tmp_path / "AAAA_0001.png"
    image = Image.new("L", (201, 51), 255)
    image.putpixel((200, 10), 0)
    image.save(source)
    with pytest.raises(ValueError, match="not pure white"):
        copy_normalized_captcha(source, tmp_path / "output.png")
