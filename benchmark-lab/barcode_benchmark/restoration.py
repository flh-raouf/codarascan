from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class TinyQRRestorer(nn.Module):
    """Small U-Net for aligned noisy-to-clean QR reconstruction."""

    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.encoder1 = ConvBlock(1, width)
        self.encoder2 = ConvBlock(width, width * 2)
        self.bottleneck = ConvBlock(width * 2, width * 4)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.decoder2 = ConvBlock(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.decoder1 = ConvBlock(width * 2, width)
        self.output = nn.Conv2d(width, 1, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        encoded1 = self.encoder1(value)
        encoded2 = self.encoder2(self.pool(encoded1))
        bottleneck = self.bottleneck(self.pool(encoded2))
        decoded2 = self.decoder2(torch.cat([self.up2(bottleneck), encoded2], dim=1))
        decoded1 = self.decoder1(torch.cat([self.up1(decoded2), encoded1], dim=1))
        return self.output(decoded1)
