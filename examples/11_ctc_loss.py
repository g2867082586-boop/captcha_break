import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from captcha_break.config import CaptchaConfig
from captcha_break.data import CaptchaDataset
from captcha_break.models import CTCRecognizer


def main() -> None:
    config = CaptchaConfig()

    dataset = CaptchaDataset(
        config=config,
        size=3,
        task="ctc",
    )

    loader = DataLoader(
        dataset,
        batch_size=3,
        num_workers=0,
    )

    images, targets = next(iter(loader))

    model = CTCRecognizer(
        n_classes=len(config.ctc_characters),
        input_shape=(
            3,
            config.height,
            config.width,
        ),
    )

    logits = model(images)

    log_probabilities = logits.log_softmax(
        dim=-1
    )

    input_lengths = torch.full(
        size=(targets.shape[0],),
        fill_value=logits.shape[0],
        dtype=torch.long,
    )

    target_lengths = torch.full(
        size=(targets.shape[0],),
        fill_value=targets.shape[1],
        dtype=torch.long,
    )

    loss = F.ctc_loss(
        log_probabilities,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True,
    )

    print("logits 形状：", logits.shape)
    print(
        "log probabilities 形状：",
        log_probabilities.shape,
    )
    print("targets 形状：", targets.shape)
    print("input lengths：", input_lengths)
    print("target lengths：", target_lengths)
    print("CTC loss：", loss.item())


if __name__ == "__main__":
    main()