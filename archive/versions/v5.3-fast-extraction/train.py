#!/usr/bin/env python3
"""Train the fast locator from generated, license-safe barcode scenes."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import zxingcpp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import FastBarcodeLocator


LINEAR_FORMATS = (
    zxingcpp.BarcodeFormat.Code39,
    zxingcpp.BarcodeFormat.Code128,
    zxingcpp.BarcodeFormat.EAN13,
    zxingcpp.BarcodeFormat.EAN8,
)
MATRIX_FORMATS = (
    zxingcpp.BarcodeFormat.QRCode,
    zxingcpp.BarcodeFormat.DataMatrix,
    zxingcpp.BarcodeFormat.PDF417,
    zxingcpp.BarcodeFormat.Aztec,
)
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _ean13(generator: random.Random) -> str:
    digits = [generator.randrange(10) for _ in range(12)]
    checksum = (
        10
        - (
            sum(digits[::2])
            + 3 * sum(digits[1::2])
        )
        % 10
    ) % 10
    return "".join(str(value) for value in (*digits, checksum))


def _payload(generator: random.Random, barcode_format: zxingcpp.BarcodeFormat) -> str:
    if barcode_format == zxingcpp.BarcodeFormat.EAN13:
        return _ean13(generator)
    if barcode_format == zxingcpp.BarcodeFormat.EAN8:
        digits = [generator.randrange(10) for _ in range(7)]
        checksum = (
            10
            - (
                3 * sum(digits[::2])
                + sum(digits[1::2])
            )
            % 10
        ) % 10
        return "".join(str(value) for value in (*digits, checksum))
    length = generator.randint(6, 24 if barcode_format in LINEAR_FORMATS else 80)
    return "".join(generator.choice(ALPHABET) for _ in range(length))


def _templates(seed: int, count: int = 512) -> list[tuple[np.ndarray, int]]:
    generator = random.Random(seed)
    output: list[tuple[np.ndarray, int]] = []
    formats = (*LINEAR_FORMATS, *MATRIX_FORMATS)
    while len(output) < count:
        barcode_format = generator.choice(formats)
        try:
            barcode = zxingcpp.create_barcode(
                _payload(generator, barcode_format),
                barcode_format,
            )
            image = np.asarray(
                barcode.to_image(scale=3, add_hrt=False, add_quiet_zones=True),
                dtype=np.uint8,
            )
        except Exception:
            continue
        if image.ndim != 2 or min(image.shape) < 8:
            continue
        output.append((image, int(barcode_format in MATRIX_FORMATS)))
    return output


class SyntheticEasyDocuments(Dataset):
    """Clear, large symbols on varied document-like backgrounds.

    The target product explicitly serves users who know their symbols are easy.
    Training therefore models ordinary print/scan variation, rotation, modest
    perspective, and clutter, but not destructive restoration cases.
    """

    def __init__(self, samples: int, size: int, seed: int) -> None:
        self.samples = samples
        self.size = size
        self.seed = seed
        self.templates = _templates(seed)

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = random.Random(self.seed + index * 1_000_003 + random.randrange(1 << 24))
        size = self.size
        base = generator.randint(205, 255)
        image = np.full((size, size), base, np.float32)
        gradient = np.linspace(
            generator.uniform(-18, 18),
            generator.uniform(-18, 18),
            size,
            dtype=np.float32,
        )
        image += gradient[None, :]

        # Document clutter: text-like strokes, rules, and pale blocks.
        for _ in range(generator.randint(5, 28)):
            color = generator.randint(45, 225)
            x = generator.randrange(size)
            y = generator.randrange(size)
            width = generator.randint(8, max(9, size // 3))
            thickness = generator.randint(1, 3)
            cv2.line(
                image,
                (x, y),
                (min(size - 1, x + width), y + generator.randint(-2, 2)),
                color,
                thickness,
            )
        target = np.zeros((size, size, 2), np.uint8)

        symbol_count = 0 if generator.random() < 0.20 else generator.randint(1, 4)
        for _ in range(symbol_count):
            patch, kind = generator.choice(self.templates)
            desired_long = generator.randint(int(size * 0.20), int(size * 0.62))
            scale = desired_long / max(patch.shape)
            resized = cv2.resize(
                patch,
                (
                    max(8, int(round(patch.shape[1] * scale))),
                    max(8, int(round(patch.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST,
            )
            patch_height, patch_width = resized.shape
            source = np.asarray(
                [
                    [0, 0],
                    [patch_width - 1, 0],
                    [patch_width - 1, patch_height - 1],
                    [0, patch_height - 1],
                ],
                np.float32,
            )
            angle = np.deg2rad(generator.uniform(-180, 180))
            cosine, sine = np.cos(angle), np.sin(angle)
            corners = source - source.mean(axis=0)
            rotation = np.asarray([[cosine, -sine], [sine, cosine]], np.float32)
            destination = corners @ rotation.T
            destination[:, 0] += generator.uniform(size * 0.16, size * 0.84)
            destination[:, 1] += generator.uniform(size * 0.16, size * 0.84)
            jitter = min(patch_height, patch_width) * generator.uniform(0.0, 0.06)
            destination += np.asarray(
                [
                    [generator.uniform(-jitter, jitter), generator.uniform(-jitter, jitter)]
                    for _ in range(4)
                ],
                np.float32,
            )
            transform = cv2.getPerspectiveTransform(source, destination)
            warped = cv2.warpPerspective(
                resized,
                transform,
                (size, size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            coverage = cv2.warpPerspective(
                np.full_like(resized, 255),
                transform,
                (size, size),
                flags=cv2.INTER_NEAREST,
            )
            image[coverage > 0] = warped[coverage > 0]
            channel = target[:, :, kind].copy()
            cv2.fillPoly(channel, [np.rint(destination).astype(np.int32)], 255)
            target[:, :, kind] = channel

        if generator.random() < 0.35:
            image = cv2.GaussianBlur(image, (0, 0), generator.uniform(0.15, 0.9))
        gain = generator.uniform(0.80, 1.18)
        bias = generator.uniform(-14, 14)
        image = np.clip(image * gain + bias, 0, 255).astype(np.uint8)
        return (
            torch.from_numpy(image[None].copy()).float().div(255.0),
            torch.from_numpy(target.transpose(2, 0, 1).copy()).float().div(255.0),
        )


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    numerator = 2 * (probability * target).sum(dim=(0, 2, 3)) + 1
    denominator = (probability + target).sum(dim=(0, 2, 3)) + 1
    return 1 - (numerator / denominator).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--script-output", type=Path)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=53_021)
    parser.add_argument("--device", default="mps")
    arguments = parser.parse_args()

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    dataset = SyntheticEasyDocuments(arguments.samples, arguments.size, arguments.seed)
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
    )
    device = torch.device(arguments.device)
    model = FastBarcodeLocator().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([4.0, 5.0], device=device).view(1, 2, 1, 1),
    )
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        total = 0.0
        for step, (images, masks) in enumerate(loader, start=1):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        print(
            f"epoch={epoch}/{arguments.epochs} loss={total / len(loader):.6f}",
            flush=True,
        )

    model = model.to("cpu").eval()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": "FastBarcodeLocator-v1",
            "training": {
                "source": "generated-zxing-scenes-only",
                "samples_per_epoch": arguments.samples,
                "epochs": arguments.epochs,
                "size": arguments.size,
                "seed": arguments.seed,
            },
        },
        arguments.output,
    )
    if arguments.script_output:
        arguments.script_output.parent.mkdir(parents=True, exist_ok=True)
        scripted = torch.jit.script(model)
        torch.jit.save(torch.jit.optimize_for_inference(scripted), arguments.script_output)
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
