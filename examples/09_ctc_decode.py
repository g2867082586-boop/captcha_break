from captcha_break.codec import ctc_greedy_decode
from captcha_break.config import CaptchaConfig


def main() -> None:
    config = CaptchaConfig()

    characters = config.ctc_characters

    print("CNN 字符数量：", len(config.alphabet))
    print("CTC 字符数量：", len(characters))
    print("CTC 字符集：", characters)

    # -AA-77-KK-22
    predicted_indices = [
        0,
        11,
        11,
        0,
        8,
        8,
        0,
        21,
        21,
        0,
        3,
        3,
    ]

    raw_text = "".join(
        characters[index]
        for index in predicted_indices
    )

    decoded_text = ctc_greedy_decode(
        predicted_indices,
        characters,
    )

    print("模型原始输出：", raw_text)
    print("CTC 解码结果：", decoded_text)


if __name__ == "__main__":
    main()