"""Encode and decode fixed-length and CTC labels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def encode_text(text: str, characters: str) -> list[int]:
    """Map every character in ``text`` to its numeric class index."""

    lookup = {character: index for index, character in enumerate(characters)}
    try:
        return [lookup[character] for character in text]
    except KeyError as error:
        raise ValueError(f"unknown character: {error.args[0]!r}") from error


def decode_indices(indices: Iterable[int], characters: str) -> str:
    """Map class indices back to a string."""

    return "".join(characters[int(index)] for index in indices)


def ctc_greedy_decode(
    indices: Sequence[int] | Iterable[int],
    characters: str,
    *,
    blank_index: int = 0,
) -> str:
    """Collapse repeated CTC predictions and remove blank tokens."""

    result: list[str] = []
    previous: int | None = None
    for raw_index in indices:
        index = int(raw_index)
        if index != previous and index != blank_index:
            result.append(characters[index])
        previous = index
    return "".join(result)
