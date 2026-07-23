#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import qrcode
import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from barcode_benchmark.io import load_json
from barcode_benchmark.tiny_locator import TinyBarcodeLocator


class ManifestSegmentationDataset(Dataset):
    def __init__(self, manifests: list[Path], size: int, seed: int) -> None:
        self.size = size
        self.seed = seed
        self.records = [
            record
            for manifest in manifests
            for record in load_json(manifest)["images"]
            if record.get("objects")
        ]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot read {record['path']}")
        height, width = image.shape[:2]
        mask = np.zeros((height, width, 2), dtype=np.uint8)
        for item in record["objects"]:
            channel = 1 if item.get("kind") == "2d" else 0
            points = np.rint(np.asarray(item["polygon"], np.float32)).astype(np.int32)
            channel_mask = mask[:, :, channel].copy()
            cv2.fillPoly(channel_mask, [points], 255)
            mask[:, :, channel] = channel_mask

        generator = random.Random(self.seed + index * 104729 + random.randrange(1 << 24))
        if generator.random() < 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        rotations = generator.randrange(4)
        image = np.rot90(image, rotations).copy()
        mask = np.rot90(mask, rotations).copy()
        if generator.random() < 0.35:
            sigma = generator.uniform(0.4, 2.2)
            image = cv2.GaussianBlur(image, (0, 0), sigma)
        gain = generator.uniform(0.72, 1.28)
        bias = generator.uniform(-24.0, 24.0)
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

        height, width = image.shape[:2]
        scale = self.size / max(height, width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        target = np.zeros((self.size, self.size, 2), dtype=np.uint8)
        canvas[:resized_height, :resized_width] = image
        target[:resized_height, :resized_width] = mask
        return (
            torch.from_numpy(canvas[:, :, ::-1].copy().transpose(2, 0, 1)).float() / 255.0,
            torch.from_numpy(target.transpose(2, 0, 1).copy()).float() / 255.0,
        )


EAN_L = (
    "0001101", "0011001", "0010011", "0111101", "0100011",
    "0110001", "0101111", "0111011", "0110111", "0001011",
)
EAN_G = (
    "0100111", "0110011", "0011011", "0100001", "0011101",
    "0111001", "0000101", "0010001", "0001001", "0010111",
)
EAN_R = tuple("".join("1" if value == "0" else "0" for value in pattern) for pattern in EAN_L)
EAN_PARITY = (
    "LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG",
    "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL",
)


def _ean13_patch(generator: random.Random) -> np.ndarray:
    digits = [generator.randrange(10) for _ in range(12)]
    checksum = (10 - (
        sum(digits[::2]) + 3 * sum(digits[1::2])
    ) % 10) % 10
    digits.append(checksum)
    left = "".join(
        (EAN_L if parity == "L" else EAN_G)[digit]
        for parity, digit in zip(EAN_PARITY[digits[0]], digits[1:7])
    )
    right = "".join(EAN_R[digit] for digit in digits[7:])
    bits = "0" * 11 + "101" + left + "01010" + right + "101" + "0" * 11
    module = generator.randint(2, 5)
    height = generator.randint(55, 120)
    patch = np.full((height, len(bits) * module), 255, np.uint8)
    for index, bit in enumerate(bits):
        if bit == "1":
            patch[: int(height * 0.82), index * module:(index + 1) * module] = 0
    return patch


def _qr_patch(generator: random.Random) -> np.ndarray:
    value = "".join(
        generator.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
        for _ in range(generator.randint(8, 80))
    )
    code = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=3,
        border=4,
    )
    code.add_data(value)
    code.make(fit=True)
    return np.asarray(code.make_image(fill_color="black", back_color="white"), np.uint8) * 255


class SyntheticBarcodeDataset(Dataset):
    """On-the-fly SBD-style multi-scale scenes without storing generated data."""

    def __init__(self, samples: int, size: int, seed: int) -> None:
        self.samples = samples
        self.size = size
        self.seed = seed

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = random.Random(self.seed + index * 130363 + random.randrange(1 << 24))
        size = self.size
        base = generator.randint(130, 245)
        image = np.full((size, size), base, np.float32)
        image += np.linspace(
            generator.uniform(-45, 45),
            generator.uniform(-45, 45),
            size,
            dtype=np.float32,
        )[None, :]
        for _ in range(generator.randint(8, 35)):
            color = generator.randint(20, 245)
            first = (generator.randrange(size), generator.randrange(size))
            second = (generator.randrange(size), generator.randrange(size))
            cv2.line(image, first, second, color, generator.randint(1, 5))
        target = np.zeros((size, size, 2), np.uint8)

        for _ in range(generator.randint(1, 5)):
            kind = 1 if generator.random() < 0.48 else 0
            patch = _qr_patch(generator) if kind else _ean13_patch(generator)
            desired_long = generator.randint(
                max(14, size // 24),
                max(28, int(size * (0.24 if kind else 0.55))),
            )
            scale = desired_long / max(patch.shape)
            patch = cv2.resize(
                patch,
                (
                    max(8, int(round(patch.shape[1] * scale))),
                    max(8, int(round(patch.shape[0] * scale))),
                ),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST,
            )
            patch_height, patch_width = patch.shape
            source = np.asarray(
                [[0, 0], [patch_width - 1, 0], [patch_width - 1, patch_height - 1], [0, patch_height - 1]],
                np.float32,
            )
            center_x = generator.uniform(0.12 * size, 0.88 * size)
            center_y = generator.uniform(0.12 * size, 0.88 * size)
            angle = np.deg2rad(generator.uniform(-180, 180))
            cosine, sine = np.cos(angle), np.sin(angle)
            corners = source - source.mean(axis=0)
            rotation = np.asarray([[cosine, -sine], [sine, cosine]], np.float32)
            destination = corners @ rotation.T
            destination[:, 0] += center_x
            destination[:, 1] += center_y
            jitter = 0.10 * min(patch_height, patch_width)
            destination += np.asarray([
                [generator.uniform(-jitter, jitter), generator.uniform(-jitter, jitter)]
                for _ in range(4)
            ], np.float32)
            transform = cv2.getPerspectiveTransform(source, destination)
            warped = cv2.warpPerspective(
                patch,
                transform,
                (size, size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            coverage = cv2.warpPerspective(
                np.full_like(patch, 255),
                transform,
                (size, size),
                flags=cv2.INTER_NEAREST,
            )
            image[coverage > 0] = warped[coverage > 0]
            channel = target[:, :, kind].copy()
            cv2.fillPoly(
                channel,
                [np.rint(destination).astype(np.int32)],
                255,
            )
            target[:, :, kind] = channel

        if generator.random() < 0.65:
            image = cv2.GaussianBlur(
                image,
                (0, 0),
                generator.uniform(0.2, 1.8),
            )
        noise = np.random.default_rng(generator.randrange(1 << 32)).normal(
            0,
            generator.uniform(0, 13),
            image.shape,
        )
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return (
            torch.from_numpy(bgr[:, :, ::-1].copy().transpose(2, 0, 1)).float() / 255.0,
            torch.from_numpy(target.transpose(2, 0, 1).copy()).float() / 255.0,
        )


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum(dim=(0, 2, 3)) + 1.0
    denominator = (probability + target).sum(dim=(0, 2, 3)) + 1.0
    return 1.0 - (numerator / denominator).mean()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the independent tiny barcode locator")
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--synthetic-samples", type=int, default=0)
    arguments = parser.parse_args()
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    real_dataset = ManifestSegmentationDataset(
        [path.resolve() for path in arguments.manifests],
        arguments.size,
        arguments.seed,
    )
    dataset: Dataset = real_dataset
    if arguments.synthetic_samples > 0:
        dataset = ConcatDataset([
            real_dataset,
            SyntheticBarcodeDataset(
                arguments.synthetic_samples,
                arguments.size,
                arguments.seed + 1_000_003,
            ),
        ])
    loader = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=True, num_workers=0)
    device = torch.device(arguments.device)
    model = TinyBarcodeLocator().to(device)
    prior_epochs = 0
    if arguments.resume:
        checkpoint = torch.load(arguments.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        prior_epochs = int(checkpoint.get("training", {}).get("epochs", 0))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=1e-4,
    )
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
            if step % 50 == 0:
                print(
                    f"epoch={epoch}/{arguments.epochs} step={step}/{len(loader)} "
                    f"loss={total / step:.6f}",
                    flush=True,
                )
        print(f"epoch={epoch}/{arguments.epochs} mean_loss={total / len(loader):.6f}", flush=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.to("cpu").state_dict(),
        "architecture": "TinyBarcodeLocator",
        "width": 16,
        "training": {
            "manifests": [str(path.resolve()) for path in arguments.manifests],
            "images": len(dataset),
            "real_images": len(real_dataset),
            "synthetic_samples_per_epoch": arguments.synthetic_samples,
            "epochs": arguments.epochs,
            "prior_epochs": prior_epochs,
            "total_epochs": prior_epochs + arguments.epochs,
            "size": arguments.size,
            "seed": arguments.seed,
            "learning_rate": arguments.learning_rate,
        },
    }, arguments.output)
    print(arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
