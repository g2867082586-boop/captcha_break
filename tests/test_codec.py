import pytest

from captcha_break.codec import ctc_greedy_decode, decode_indices, encode_text


def test_encode_and_decode_round_trip() -> None:
    characters = "ABC123"
    encoded = encode_text("A2C", characters)
    assert encoded == [0, 4, 2]
    assert decode_indices(encoded, characters) == "A2C"


def test_unknown_character_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown character"):
        encode_text("Z", "ABC")


def test_ctc_decode_removes_blanks_and_repeats() -> None:
    characters = "-ABC"
    assert ctc_greedy_decode([0, 1, 1, 0, 2, 2, 3], characters) == "ABC"
