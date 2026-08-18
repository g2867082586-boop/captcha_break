import pytest

torch = pytest.importorskip("torch")

from captcha_break.models import CTCRecognizer, FixedLengthCNN


def test_fixed_length_cnn_output_shape() -> None:
    model = FixedLengthCNN(n_classes=36, label_length=4).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 192))
    assert output.shape == (2, 4, 36)


def test_ctc_output_shape() -> None:
    model = CTCRecognizer(n_classes=37, input_shape=(3, 64, 192)).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 3, 64, 192))
    assert output.shape == (12, 2, 37)
