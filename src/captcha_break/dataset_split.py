"""Deterministic, balanced splitting for labeled real CAPTCHA datasets."""

from __future__ import annotations

import hashlib
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .data import label_from_filename
from .project_generator import VISUAL_STYLES, VisualStyle


@dataclass(frozen=True, slots=True)
class ThreeWaySplit:
    train: tuple[Path, ...]
    validation: tuple[Path, ...]
    test: tuple[Path, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def character_counts(paths: list[Path] | tuple[Path, ...], alphabet: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for path in paths:
        counts.update(label_from_filename(path, alphabet))
    return counts


def _style_quotas(
    paths: list[Path], styles: dict[Path, VisualStyle], selection_size: int
) -> dict[VisualStyle, int]:
    counts = Counter(styles[path] for path in paths)
    raw = {style: counts[style] * selection_size / len(paths) for style in VISUAL_STYLES}
    quotas = {style: int(raw[style]) for style in VISUAL_STYLES}
    remaining = selection_size - sum(quotas.values())
    ranked = sorted(
        VISUAL_STYLES,
        key=lambda style: (raw[style] - quotas[style], counts[style]),
        reverse=True,
    )
    for style in ranked:
        if remaining == 0:
            break
        if quotas[style] < counts[style]:
            quotas[style] += 1
            remaining -= 1
    if remaining:
        raise ValueError("could not allocate the requested style quotas")
    return quotas


def choose_balanced_subset(
    paths: list[Path],
    styles: dict[Path, VisualStyle],
    *,
    size: int,
    alphabet: str,
    seed: int,
    candidates: int = 5000,
) -> tuple[Path, ...]:
    if not 0 < size < len(paths):
        raise ValueError("subset size must be between zero and the number of paths")
    if candidates <= 0:
        raise ValueError("candidates must be positive")
    quotas = _style_quotas(paths, styles, size)
    groups: dict[VisualStyle, list[Path]] = defaultdict(list)
    for path in paths:
        groups[styles[path]].append(path)

    full_counts = character_counts(paths, alphabet)
    expected = {character: full_counts[character] * size / len(paths) for character in alphabet}
    best_paths: tuple[Path, ...] = ()
    best_score: tuple[int, int, int, float] | None = None
    for attempt in range(candidates):
        selected: list[Path] = []
        for style_index, style in enumerate(VISUAL_STYLES):
            group = groups[style].copy()
            random.Random(seed + attempt * len(VISUAL_STYLES) + style_index).shuffle(group)
            selected.extend(group[: quotas[style]])
        counts = character_counts(selected, alphabet)
        covered = sum(counts[character] > 0 for character in alphabet)
        minimum = min(counts[character] for character in alphabet)
        capped_balance = sum(min(counts[character], 5) for character in alphabet)
        distribution_error = sum(
            abs(counts[character] - expected[character]) for character in alphabet
        )
        score = (covered, minimum, capped_balance, -distribution_error)
        if best_score is None or score > best_score:
            best_score = score
            best_paths = tuple(sorted(selected))
    return best_paths


def make_three_way_split(
    paths: list[Path],
    styles: dict[Path, VisualStyle],
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    alphabet: str,
    seed: int,
    candidates: int = 5000,
) -> ThreeWaySplit:
    if train_size + validation_size + test_size != len(paths):
        raise ValueError("split sizes must add up to the number of images")
    if min(train_size, validation_size, test_size) <= 0:
        raise ValueError("all split sizes must be positive")

    test = choose_balanced_subset(
        paths,
        styles,
        size=test_size,
        alphabet=alphabet,
        seed=seed,
        candidates=candidates,
    )
    test_set = set(test)
    remaining = [path for path in paths if path not in test_set]
    validation = choose_balanced_subset(
        remaining,
        styles,
        size=validation_size,
        alphabet=alphabet,
        seed=seed + 1,
        candidates=candidates,
    )
    validation_set = set(validation)
    train = tuple(sorted(path for path in remaining if path not in validation_set))
    if len(train) != train_size:
        raise RuntimeError("training split size does not match")
    return ThreeWaySplit(train=train, validation=validation, test=test)


def copy_normalized_captcha(
    source: Path,
    destination: Path,
    *,
    expected_size: tuple[int, int] = (200, 50),
) -> str:
    """Copy an exact-size image or remove the known one-pixel white right/bottom pad."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        if image.size == expected_size:
            shutil.copy2(source, destination)
            return "copied"
        padded_size = (expected_size[0] + 1, expected_size[1] + 1)
        if image.size != padded_size:
            raise ValueError(f"unsupported image size {image.size}: {source.name}")
        grayscale = image.convert("L")
        right_edge = grayscale.crop((expected_size[0], 0, padded_size[0], padded_size[1]))
        bottom_edge = grayscale.crop((0, expected_size[1], padded_size[0], padded_size[1]))
        if right_edge.getextrema() != (255, 255) or bottom_edge.getextrema() != (255, 255):
            raise ValueError(f"extra image edges are not pure white: {source.name}")
        image.crop((0, 0, expected_size[0], expected_size[1])).save(destination)
    return "cropped"
