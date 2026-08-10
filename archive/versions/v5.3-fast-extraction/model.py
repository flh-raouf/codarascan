"""Small CPU-oriented segmentation model for clear, large barcodes.

This is an independently authored architecture.  It deliberately uses a
single grayscale input, three early strided convolutions, one low-resolution
context branch, and one pixel-shuffle output.  The design minimizes full-size
feature maps and interpolation calls because actual CPU latency matters more
than parameter count alone.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )


class FastBarcodeLocator(nn.Module):
    """Two-class (linear/matrix) segmentation at one-eighth resolution."""

    def __init__(self, classes: int = 2) -> None:
        super().__init__()
        self.downsample = nn.Sequential(
            ConvNormAct(1, 8, stride=2),
            ConvNormAct(8, 12, stride=2),
            ConvNormAct(12, 16, stride=2),
        )
        self.context = nn.Sequential(
            ConvNormAct(16, 24, stride=2),
            ConvNormAct(24, 24),
            ConvNormAct(24, 24, stride=2),
            ConvNormAct(24, 24),
        )
        self.high_projection = ConvNormAct(16, 24, kernel_size=1)
        self.low_projection = ConvNormAct(24, 24, kernel_size=1)
        self.refine = ConvNormAct(24, 24)
        self.classifier = nn.Conv2d(24, classes * 64, 1)
        self.pixel_shuffle = nn.PixelShuffle(8)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        high = self.downsample(value)
        low = self.context(high)
        low = F.interpolate(
            self.low_projection(low),
            size=high.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        fused = self.refine(self.high_projection(high) + low)
        return self.pixel_shuffle(self.classifier(fused))
