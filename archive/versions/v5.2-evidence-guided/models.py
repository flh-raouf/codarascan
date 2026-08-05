"""Small, locally authored neural components used by the 5.2 pipeline."""
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
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )


class SeparableBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = ConvNormAct(
            input_channels,
            input_channels,
            stride=stride,
            groups=input_channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(value))


class TinyBarcodeLocator(nn.Module):
    """424 KB two-class full-frame locator."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.stem = ConvNormAct(3, width, stride=2)
        self.stage4 = SeparableBlock(width, width * 2, stride=2)
        self.stage8 = nn.Sequential(
            SeparableBlock(width * 2, width * 3, stride=2),
            SeparableBlock(width * 3, width * 3),
        )
        self.stage16 = nn.Sequential(
            SeparableBlock(width * 3, width * 4, stride=2),
            SeparableBlock(width * 4, width * 4),
            SeparableBlock(width * 4, width * 4),
        )
        self.fuse8 = nn.Sequential(
            ConvNormAct(width * 7, width * 3),
            SeparableBlock(width * 3, width * 3),
        )
        self.fuse4 = nn.Sequential(
            ConvNormAct(width * 5, width * 2),
            SeparableBlock(width * 2, width * 2),
        )
        self.classifier = nn.Conv2d(width * 2, 2, 1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.stem(value)
        quarter = self.stage4(half)
        eighth = self.stage8(quarter)
        sixteenth = self.stage16(eighth)
        decoded8 = F.interpolate(
            sixteenth,
            size=eighth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded8 = self.fuse8(torch.cat([decoded8, eighth], dim=1))
        decoded4 = F.interpolate(
            decoded8,
            size=quarter.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded4 = self.fuse4(torch.cat([decoded4, quarter], dim=1))
        return F.interpolate(
            self.classifier(decoded4),
            size=value.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class TinyQRRestorer(nn.Module):
    """124 KB U-Net trained on aligned QRDN noisy/clean pairs."""

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
