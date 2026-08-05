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
    """Permissively authored compact two-class segmentation locator."""

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
        logits = self.classifier(decoded4)
        return F.interpolate(
            logits,
            size=value.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
