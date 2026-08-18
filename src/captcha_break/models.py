"""Modern PyTorch versions of the original CNN and CNN-RNN-CTC models."""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn


def _vgg_features(pools: list[int | tuple[int, int]]) -> nn.Sequential:
    channels = [32, 64, 128, 256, 256]
    modules: OrderedDict[str, nn.Module] = OrderedDict()
    in_channels = 3
    for block_index, (out_channels, pool) in enumerate(zip(channels, pools), start=1):
        for layer_index in range(1, 3):
            name = f"{block_index}_{layer_index}"
            modules[f"conv_{name}"] = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            modules[f"batch_norm_{name}"] = nn.BatchNorm2d(out_channels)
            modules[f"relu_{name}"] = nn.ReLU(inplace=True)
            in_channels = out_channels
        modules[f"pool_{block_index}"] = nn.MaxPool2d(pool)
    return nn.Sequential(modules)


class FixedLengthCNN(nn.Module):
    """VGG-style recognizer with one classifier per character position."""

    def __init__(self, n_classes: int, label_length: int = 4) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.label_length = label_length
        self.features = _vgg_features([2, 2, 2, 2, 2])
        self.pool = nn.AdaptiveAvgPool2d((2, 4))
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(256 * 2 * 4, label_length * n_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.pool(self.features(images))
        features = self.dropout(torch.flatten(features, 1))
        logits = self.classifier(features)
        return logits.view(images.shape[0], self.label_length, self.n_classes)


class CTCRecognizer(nn.Module):
    """CNN feature extractor followed by a bidirectional LSTM and CTC head."""

    def __init__(
        self,
        n_classes: int,
        input_shape: tuple[int, int, int] = (3, 64, 192),
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.features = _vgg_features([2, 2, 2, 2, (2, 1)])
        feature_size = self._infer_feature_size()
        self.dropout = nn.Dropout(0.25)
        self.rnn = nn.LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, n_classes)

    def _infer_feature_size(self) -> int:
        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            sample = torch.zeros((1, *self.input_shape))
            output = self.features(sample)
        self.features.train(was_training)
        return output.shape[1] * output.shape[2]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.dropout(self.features(images))
        # B,C,H,W -> W,B,C*H: width becomes the sequence/time dimension.
        sequence = features.permute(3, 0, 1, 2).flatten(2)
        sequence, _ = self.rnn(sequence)
        return self.classifier(sequence)
