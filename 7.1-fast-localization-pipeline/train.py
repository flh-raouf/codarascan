#!/usr/bin/env python3
"""Fine-tune the clean 31k-parameter locator on permitted development data."""

from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


SCRIPT_DIR = Path(__file__).resolve().parent
LAB_DIR = SCRIPT_DIR.parent / "benchmark-lab"
MODEL_SOURCE = SCRIPT_DIR.parent / "5.3-fast-extraction-pipeline" / "model.py"
BASE_CHECKPOINT = (
    SCRIPT_DIR.parent
    / "5.3-fast-extraction-pipeline"
    / "models"
    / "fast-locator-v1.pt"
)
if str(LAB_DIR) not in sys.path:
    sys.path.insert(0, str(LAB_DIR))

from barcode_benchmark.io import load_json  # noqa: E402


def load_model_class() -> type[nn.Module]:
    spec = importlib.util.spec_from_file_location(
        "_fast_locator_model",
        MODEL_SOURCE,
    )
    if spec is None or spec.loader is None:
        raise ImportError(MODEL_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.FastBarcodeLocator


class ManifestPages(Dataset):
    def __init__(
        self,
        manifests: list[Path],
        *,
        size: int,
        seed: int,
        repeats: int,
    ) -> None:
        self.records = [
            record
            for manifest in manifests
            for record in load_json(manifest)["images"]
        ]
        self.size = size
        self.seed = seed
        self.repeats = repeats
        self.cached: list[tuple[np.ndarray, np.ndarray]] = [
            self._prepare(record) for record in self.records
        ]

    def _prepare(
        self,
        record: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        image = cv2.imread(str(record["path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(record["path"])
        height, width = image.shape
        target = np.zeros((height, width, 2), np.uint8)
        for item in record.get("objects", []):
            channel = 1 if item.get("kind") == "2d" else 0
            polygon = np.rint(
                np.asarray(item["polygon"], np.float32)
            ).astype(np.int32)
            channel_mask = target[:, :, channel].copy()
            cv2.fillPoly(channel_mask, [polygon], 255)
            target[:, :, channel] = channel_mask
        scale = self.size / max(height, width)
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        target = cv2.resize(
            target,
            (resized_width, resized_height),
            interpolation=cv2.INTER_NEAREST,
        )
        canvas = np.full((self.size, self.size), 255, np.uint8)
        mask = np.zeros((self.size, self.size, 2), np.uint8)
        canvas[:resized_height, :resized_width] = image
        mask[:resized_height, :resized_width] = target
        return canvas, mask

    def __len__(self) -> int:
        return len(self.records) * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record_index = index % len(self.records)
        repetition = index // len(self.records)
        image, target = self.cached[record_index]
        image = image.copy()
        target = target.copy()

        generator = random.Random(
            self.seed
            + record_index * 1_000_003
            + repetition * 104_729
            + random.randrange(1 << 24)
        )
        if generator.random() < 0.5:
            image = cv2.flip(image, 1)
            target = cv2.flip(target, 1)
        rotations = generator.randrange(4)
        image = np.rot90(image, rotations).copy()
        target = np.rot90(target, rotations).copy()
        if generator.random() < 0.30:
            image = cv2.GaussianBlur(
                image,
                (0, 0),
                generator.uniform(0.2, 1.4),
            )
        image = np.clip(
            image.astype(np.float32) * generator.uniform(0.80, 1.20)
            + generator.uniform(-18, 18),
            0,
            255,
        ).astype(np.uint8)

        return (
            torch.from_numpy(image[None].copy()).float().div(255.0),
            torch.from_numpy(target.transpose(2, 0, 1).copy()).float().div(255.0),
        )


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).sum((0, 2, 3)) + 1.0
    denominator = (probability + target).sum((0, 2, 3)) + 1.0
    return 1.0 - (numerator / denominator).mean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--script-output", required=True, type=Path)
    parser.add_argument("--base", type=Path, default=BASE_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=71_071)
    parser.add_argument("--device", default="mps")
    arguments = parser.parse_args()

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    dataset = ManifestPages(
        [path.resolve() for path in arguments.manifests],
        size=arguments.size,
        seed=arguments.seed,
        repeats=arguments.repeats,
    )
    loader = DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=0,
    )
    model_class = load_model_class()
    model = model_class()
    checkpoint = torch.load(
        arguments.base,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model"])
    device = torch.device(arguments.device)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=1e-4,
    )
    bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [4.0, 5.0],
            device=device,
        ).view(1, 2, 1, 1),
    )
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        total = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        print(
            f"epoch={epoch}/{arguments.epochs} "
            f"loss={total / len(loader):.6f}",
            flush=True,
        )

    model = model.to("cpu").eval()
    training = {
        "source": "manifest-supervised-fine-tuning",
        "manifests": [
            str(path.resolve()) for path in arguments.manifests
        ],
        "images_per_epoch": len(dataset),
        "epochs": arguments.epochs,
        "size": arguments.size,
        "seed": arguments.seed,
        "base": str(arguments.base.resolve()),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": "FastBarcodeLocator-v1",
            "training": training,
        },
        arguments.output,
    )
    arguments.script_output.parent.mkdir(parents=True, exist_ok=True)
    scripted = torch.jit.script(model)
    torch.jit.save(
        torch.jit.optimize_for_inference(scripted),
        arguments.script_output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
