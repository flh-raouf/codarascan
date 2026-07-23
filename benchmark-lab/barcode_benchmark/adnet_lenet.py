"""Checkpoint-compatible evaluator for the external ADNet LENet model.

This module exists only to benchmark the authors' released weights.  The
upstream repository has no declared license and its package initializer imports
a syntactically invalid, unrelated architecture under Python 3.13, so it is
not a candidate for the shipped pipeline.
"""
from __future__ import annotations

import torch
from torch import nn


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).square().mean(1, keepdim=True)
        normalized = (value - mean) / (variance + self.eps).sqrt()
        return (
            self.weight[None, :, None, None] * normalized
            + self.bias[None, :, None, None]
        )


class SimpleGate(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=1)
        return first * second


class LightNAFBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels * 2, 1)
        self.dw_conv = nn.Conv2d(
            channels * 2,
            channels * 2,
            3,
            padding=1,
            groups=channels * 2,
        )
        self.sg = SimpleGate()
        self.conv2 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.conv2(self.sg(self.dw_conv(self.conv1(self.norm(value)))))
        return value + residual * self.beta


class DownSample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class UpSample(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels * 4, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(value)


class ExternalADNetLENet(nn.Module):
    def __init__(self, input_channels: int = 3, base_channels: int = 16) -> None:
        super().__init__()
        base = base_channels
        self.intro = nn.Conv2d(input_channels, base, 3, padding=1)
        self.outro = nn.Conv2d(base, input_channels, 3, padding=1)
        self.enc1 = LightNAFBlock(base)
        self.down1 = DownSample(base, base * 2)
        self.enc2 = LightNAFBlock(base * 2)
        self.down2 = DownSample(base * 2, base * 4)
        self.enc3 = LightNAFBlock(base * 4)
        self.down3 = DownSample(base * 4, base * 8)
        self.bottleneck = nn.Sequential(
            LightNAFBlock(base * 8),
            LightNAFBlock(base * 8),
        )
        self.up3 = UpSample(base * 8, base * 4)
        self.dec3 = LightNAFBlock(base * 4)
        self.up2 = UpSample(base * 4, base * 2)
        self.dec2 = LightNAFBlock(base * 2)
        self.up1 = UpSample(base * 2, base)
        self.dec1 = LightNAFBlock(base)
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(base, base // 2, 1),
            nn.Conv2d(base // 2, base, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        source = value
        value = self.intro(value)
        level1 = self.enc1(value)
        level2 = self.enc2(self.down1(level1))
        level3 = self.enc3(self.down2(level2))
        value = self.bottleneck(self.down3(level3))
        value = self.dec3(self.up3(value) + level3)
        value = self.dec2(self.up2(value) + level2)
        value = self.dec1(self.up1(value) + level1)
        value = value * (1.0 + self.edge_enhance(value))
        return self.outro(value) + source
