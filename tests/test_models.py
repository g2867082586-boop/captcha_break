import pytest

torch = pytest.importorskip("torch")

from captcha_break.models import CTCRecognizer, FixedLengthCNN, ProjectCaptchaCNN


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


def test_project_cnn_output_shape() -> None:
    model = ProjectCaptchaCNN(n_classes=36, label_length=4).eval()
    with torch.no_grad():
        output = model(torch.zeros(2, 1, 50, 200))
    assert output.shape == (2, 4, 36)


def test_project_cnn_rejects_rgb_input() -> None:
    model = ProjectCaptchaCNN().eval()

    with pytest.raises(ValueError, match="one grayscale channel"):
        model(torch.zeros(2, 3, 50, 200))


def test_project_cnn_rejects_wrong_image_size() -> None:
    model = ProjectCaptchaCNN().eval()

    with pytest.raises(ValueError, match="expected spatial size"):
        model(torch.zeros(2, 1, 64, 192))


def test_project_cnn_is_smaller_than_legacy_cnn() -> None:
    project_model = ProjectCaptchaCNN()
    legacy_model = FixedLengthCNN(n_classes=36, label_length=4)

    project_parameters = sum(parameter.numel() for parameter in project_model.parameters())
    legacy_parameters = sum(parameter.numel() for parameter in legacy_model.parameters())

    assert project_parameters < legacy_parameters
